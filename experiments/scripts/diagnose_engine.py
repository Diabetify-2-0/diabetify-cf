try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError:  
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)


if __name__ == "__main__":
    from experiments.scripts.diagnose_ocean import main

    main()
