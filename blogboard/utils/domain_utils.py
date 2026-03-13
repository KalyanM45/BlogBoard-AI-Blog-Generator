from datetime import datetime
from typing import Dict
from blogboard.config.settings import app_settings

def get_current_day_domain() -> str:
    day_name = datetime.now().strftime("%A")
    
    mapping = {
        "Monday": "Machine Learning",
        "Tuesday": "Deep Learning",
        "Wednesday": "Statistics",
        "Thursday": "Natural Language Processing",
        "Friday": "Computer Vision",
        "Saturday": "Generative AI",
        "Sunday": "Recent AI News"
    }
    
    return mapping.get(day_name)


