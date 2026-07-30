from app.main.routes import main_bp
print("Blueprints registered:", [bp.name for bp in Flask().blueprints.values()])
