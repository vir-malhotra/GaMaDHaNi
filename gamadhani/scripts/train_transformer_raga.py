import wandb
import logging
import os
import pytorch_lightning as pl
import torch
from absl import app, flags
from torch.utils import data
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import gamadhani
from gamadhani.src.protobuf.data_example import AudioExample
from gamadhani.src.dataset import SequenceDataset
from gamadhani.src.model_transformer_raga import XTransformerPriorRaga
from gamadhani.utils.utils import search_for_run
from pytorch_lightning.plugins.environments import SLURMEnvironment

import wandb
import time
import pdb
import gin

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("config",
                          default="decoder_only",
                          help="config file to parse.")
flags.DEFINE_string("db_path",
                    default=None,
                    help="path to dataset.")
flags.DEFINE_integer("val_size",
                     default=8192,
                     help="size of validation dataset.")
flags.DEFINE_integer("batch_size", default=64, help="batch size.")
flags.DEFINE_string("name", default=None, required=False, help="train name.")
flags.DEFINE_integer("gpu", default=0, help="gpu index.")
flags.DEFINE_integer("workers",
                     default=4,
                     help="num workers during data loading.")
flags.DEFINE_integer("patience",
                     default=20,
                     help="patience parameter for EarlyStopping")
flags.DEFINE_string("pretrained_embedding",
                    default=None,
                    help="use pretrained embeddings from rave.")
flags.DEFINE_multi_string("override",
                          default=[],
                          help="additional gin bindings.")
flags.DEFINE_string("ckpt",
                    default=None,
                    help="checkpoint to resume training from.")
flags.DEFINE_float('ema',
                   default=None,
                   help='Exponential weight averaging factor (optional)')
flags.DEFINE_integer("val_every",
                     default=10,
                     help="validate training every n epochs.")
flags.DEFINE_integer("checkpoint_model_every",
                     default=10,
                     help="checkpoint model every n epochs.")
flags.DEFINE_string("data_art",
                    default=None,
                    help="data artifact name on wandb.")
flags.DEFINE_bool("split",
                    default=False,
                    help="If true, will randomly split the dataset into train and validation sets.")
flags.DEFINE_bool("debug",
                    default=False,
                    help="If true will run in debug mode")
flags.DEFINE_string("group",
                    default="raga-conditioning",
                    help="wandb group")
flags.DEFINE_integer("max_epochs",
                     default=1000,
                     help="Maximum number of epochs to train for.")
flags.DEFINE_string("id_to_checkpoint_from",
                    default=None,
                    help="SLURM id to continue from. If None, it is ignored.")
flags.DEFINE_string("run_name",
                    default=None,
                    help="Run name to add on wandb. If None, will default to run_id (SLURM JOB ID)")
flags.DEFINE_string("wandb_notes",
                    default=None,
                    help="Notes to add to wandb (optional).")
flags.DEFINE_bool("log_to_wandb",
                    default=None,
                    help="Option to automatically log to wandb. By default is true when debug is false else false.")
flags.DEFINE_string("wandb_id",
                    default=None,
                    help="Option to allow user to explicitly enter wandb id, in the case that wandb id != slurm id.")
flags.DEFINE_integer("num_ragas",
                     default=None,
                     help="Number of ragas in dataset. If None, will be inferred from data.")


def add_ext(config: str):
    if config[-4:] != ".gin":
        config += ".gin"
    if not os.path.exists(config):
        config = os.path.join(os.getcwd(), config)
    return config


def infer_num_ragas(dataset):
    """Infer number of unique ragas in dataset."""
    ragas = set()
    sample_size = min(1000, len(dataset))  # Sample first 1000 examples

    logging.info(f"Inferring number of ragas from {sample_size} samples...")

    for i in range(sample_size):
        try:
            sample = dataset[i]
            if 'raga' in sample and sample['raga'] >= 0:
                ragas.add(int(sample['raga']))
        except Exception as e:
            logging.warning(f"Error reading sample {i}: {e}")
            continue

    num_ragas = len(ragas) if ragas else 64  # Default to 64 if none found
    logging.info(f"Found {num_ragas} unique ragas: {sorted(ragas) if ragas else 'none detected'}")

    return num_ragas


def main(argv):

    if FLAGS.id_to_checkpoint_from is not None:
        run_id = FLAGS.id_to_checkpoint_from
    else:
        run_id = os.environ.get('SLURM_ARRAY_JOB_ID') if 'SLURM_ARRAY_JOB_ID' in os.environ else os.environ.get('SLURM_JOB_ID')
        if run_id is None:
            run_id = f"local_{int(time.time())}"
        if FLAGS.debug:
            run_id += '_' + str(int(time.time()))

    if FLAGS.wandb_id is not None:
        wandb_id = FLAGS.wandb_id
    else:
        wandb_id = run_id

    if FLAGS.run_name is not None:
        run_name = FLAGS.run_name
    else:
        run_name = run_id

    # Handle SLURM tmpdir for local/HPC compatibility
    tmp = os.environ.get('SLURM_TMPDIR', '/tmp')
    tmp = os.path.join(tmp, run_id)
    os.makedirs(tmp, exist_ok=True)

    if FLAGS.ckpt is None:
        scratch = os.environ.get('SCRATCH', os.path.expanduser('~/scratch'))
        checkpoint_folder = os.path.join(scratch, 'checkpoints', 'gamadhani_raga', FLAGS.group, run_id)
        os.makedirs(checkpoint_folder, exist_ok=True)
        FLAGS.ckpt = checkpoint_folder
    elif not os.path.exists(FLAGS.ckpt):
        raise Exception('Checkpoint path does not exist')
    else:
        checkpoint_folder = FLAGS.ckpt

    if FLAGS.log_to_wandb is None:
        if FLAGS.debug:
            log_to_wandb = False
        else:
            log_to_wandb = True
    else:
        log_to_wandb = FLAGS.log_to_wandb

    if log_to_wandb:
        wandb_run = wandb.init(
            project='gamadhani-raga-conditioning',
            dir=tmp,
            config={key: value.value for (key, value) in FLAGS.__dict__['__flags'].items()},
            job_type='train',
            resume='allow' if not FLAGS.debug else 'never',
            group=FLAGS.group,
            id=wandb_id,
            name=run_name,
            notes=FLAGS.wandb_notes
        )

        # log slurm id to wandb
        if 'SLURM ID' in wandb_run.config.keys():
            slurm_ids = wandb_run.config['SLURM ID']
            if isinstance(slurm_ids, list):
                slurm_ids.append(run_id)
            else:
                slurm_ids = [slurm_ids]
                slurm_ids.append(run_id)
            wandb_run.config.update({
                'SLURM ID': run_id
            }, allow_val_change=True)
        else:
            wandb_run.config.update({
                'SLURM ID': run_id
            })

    overrides = FLAGS.override
    if FLAGS.pretrained_embedding is not None:
        overrides.append(f"PRETRAINED_RAVE='{FLAGS.pretrained_embedding}'")

    logging.info("parsing configuration")
    configs = list(map(add_ext, FLAGS.config))
    gin.parse_config_files_and_bindings(
        configs,
        overrides,
    )

    logging.info("loading dataset")
    if FLAGS.data_art is not None:
        dataset = wandb_run.use_artifact(FLAGS.data_art + ":latest").download(root=os.path.join(tmp, 'data'))
        FLAGS.db_path = os.path.join(dataset)
    elif FLAGS.db_path is not None:
        dataset = FLAGS.db_path
    else:
        raise ValueError("Must provide either data_artifact or data")

    if FLAGS.split:
        dataset = SequenceDataset(db_path=FLAGS.db_path)

        if FLAGS.val_size > len(dataset):
            logging.warn(
                r"Dataset too small, using 5% of the train set as the val set")
            FLAGS.val_size = len(dataset) // 20

        train, val = data.random_split(
            dataset,
            (len(dataset) - FLAGS.val_size, FLAGS.val_size),
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train = SequenceDataset(db_path=os.path.join(FLAGS.db_path, 'train'))
        val = SequenceDataset(db_path=os.path.join(FLAGS.db_path, 'val'))

    # Infer number of ragas if not provided
    if FLAGS.num_ragas is not None:
        num_ragas = FLAGS.num_ragas
        logging.info(f"Using user-specified num_ragas: {num_ragas}")
    else:
        num_ragas = infer_num_ragas(train)
        logging.info(f"Inferred num_ragas: {num_ragas}")

    # Update gin config with num_ragas
    with gin.unlock_config():
        gin.parse_config(f"NUM_RAGAS={num_ragas}")

    if not any(map(lambda x: "flattened" in x, FLAGS.config)):
        logging.info("quantizer number retrieval")
        with gin.unlock_config():
            gin.parse_config(
                f"NUM_QUANTIZERS={train[0]['decoder_inputs'].shape[-1]}")

    logging.info("building model")
    model = XTransformerPriorRaga()

    logging.info(f"Model configuration:")
    logging.info(f"  Raga conditioning: {model.use_raga_conditioning}")
    logging.info(f"  Number of ragas: {model.num_ragas}")
    logging.info(f"  Raga embedding dim: {model.raga_dim}")

    train_loader = data.DataLoader(
        train,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=FLAGS.workers,
    )
    val_loader = data.DataLoader(
        val,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=FLAGS.workers,
    )

    with open(os.path.join(checkpoint_folder, "config.gin"),
              "w") as config_out:
        config_out.write(gin.config_str())

    val_check = {}
    val_check["check_val_every_n_epoch"] = FLAGS.val_every

    last_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_folder, 'models'),
        filename="checkpoint-{epoch:02d}-{val_cross_entropy:.2f}-{cross_entropy:.2f}",
        save_on_train_epoch_end=True,
        every_n_epochs=FLAGS.checkpoint_model_every,
        save_last=True,
        save_top_k=-1
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_cross_entropy",
            filename='best',
            dirpath=os.path.join(checkpoint_folder, 'models')
        ),
        last_checkpoint,
        pl.callbacks.EarlyStopping(
            "val_cross_entropy",
            patience=FLAGS.patience,
        )
    ]

    if FLAGS.ema is not None:
        callbacks.append(gamadhani.utils.EMA(FLAGS.ema))

    logging.info("creating trainer")
    if log_to_wandb:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
        wandb_logger = pl.loggers.WandbLogger(
            project="gamadhani-raga-conditioning",
            save_dir=os.path.join(tmp, f'wandb-{wandb_run.name}'),
            offline=False,
            id=wandb_run.id
        )
    else:
        wandb_logger = None

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        devices=[FLAGS.gpu],
        callbacks=callbacks,
        log_every_n_steps=10,
        **val_check,
        max_epochs=FLAGS.max_epochs,
        plugins=[SLURMEnvironment(auto_requeue=True)],
        profiler="simple"
    )

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    logging.info("launch training")
    run = search_for_run(checkpoint_folder)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=run,
    )

    if log_to_wandb:
        wandb.finish()

if __name__ == "__main__":
    app.run(main)
