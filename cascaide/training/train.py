import argparse
from cascaide.configs.config import load_config
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                          help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None,
                          help="Override resume_from in config")
    args = parser.parse_args()

    cfg, raw = load_config(args.config)
    if args.resume:
        cfg.training.resume_from = args.resume
        raw["training"]["resume_from"] = args.resume

    trainer = Trainer(cfg, raw)
    trainer.fit()


if __name__ == "__main__":
    main()