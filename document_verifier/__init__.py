from flask import Flask

from .config import Config
from .models import db
from .routes import bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    config_class.ensure_directories()

    db.init_app(app)
    app.register_blueprint(bp)

    with app.app_context():
        db.create_all()

    return app
