def choose_model(configs):
    """ 
    Ask user to choose model/config to use.
    """
    n = -1
    while n < 0 or n >= len(configs):
        print("Select model:")
        for i, config in enumerate(configs):
            print(f"[{i}] {config['name']}")
        n = int(input() or 0)

    config = configs[n]
    print("Loaded model configuration:")
    for key, value in config.items():
        print(f"\t* {key}: {value}")

    return config
