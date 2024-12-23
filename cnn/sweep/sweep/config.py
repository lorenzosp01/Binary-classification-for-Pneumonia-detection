sweep_confiuration = {
    "method": "grid",
    "metric": {
        "name": "loss",
        "goal": "minimize"
    },
    "parameters": {
        "dataset": {
            "values": ["CXr"]
        },
        "batch_size": {
            "values": [32, 64, 128]
        },
        "learning_rate": {
            "values": [0.0001],
        },
        "epochs": {
            "value": 50
        },
        "dropout": {
            "values": [0.3, 0.4]
        },
        "optimizer": {
            "values": ["adam"]
        },
        "layer_size": {
            "values": [64, 128]
        },
        "conv_size": {
            "values": [3, 6]
        }
    }
}