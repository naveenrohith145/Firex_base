import os.path

import toml

class AppConfig:
    file_path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(file_path,"config.toml")
    try:
        config=toml.load(config_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found:{config_path}")
    db_configuration=config["database"]
    
    

    print("setting db configuration")
    SQLALCHEMY_DATABASE_URI=(
        f"postgresql://{db_configuration['user']}:{db_configuration['password']}@{db_configuration['host']}:{db_configuration['port']}/{db_configuration['dbname']}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False