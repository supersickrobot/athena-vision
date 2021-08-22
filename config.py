def configuration():
    config = {
        "verbosity": 3,
        "web": {
            "host": "192.168.1.9",
            "port": 5055
        },
        "comm": {
            "url": "http://192.168.1.9:8000"
        },
        "safety":{
            "device_id": "f0265103",
            "depth": {
                "resolution": [320, 240],
                "scale": 0.001,
                "fps": 30
            },
            "color": {
                "resolution": [640, 480],
                "fps": 30
            },
        },
        "camera": {
            "device_id": "f0221610",
            "alpha": 0.5,
            "edge": {
                "low_threshold": 50,
                "ratio": 2,
                "kernal": 3
            },
            "depth": {
                "resolution": [1024, 768],
                "scale": 0.001,
                "fps": 30
            },
            "color": {
                "resolution": [1280, 720],
                "fps": 30
            },
            "preset": "None",
            "initialization": {
                "timer": 30,  # num of frames
                "bins": 200,
                "camera_height": 10000,
                "mult": 3
            },
            "object_identification": {
                "contour": {
                    "low": 100,
                    "high": 255
                },
                "area": {
                    "low": 900,
                    "high": 100000
                },
                "background": {
                    "bins": 100,
                    "mult": 2
                },
            "depth_random_sample_size" : 30

            }
        }
    }
    return config