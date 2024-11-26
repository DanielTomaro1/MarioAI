# utils.py
def display_status(info_dict):
    status = "\n=== Mario AI Training Status ===\n"
    for key, value in info_dict.items():
        status += f"{key}: {value}\n"
    status += "\nPress Ctrl+C to save and exit\n"
    print(status)
