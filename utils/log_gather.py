import os

def gather_logs(folder_path, output_filename="log_report.log"):
    output_path = os.path.join(folder_path, output_filename)
    with open(output_path, "w", encoding="utf-8") as out_file:
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname.endswith(".log") and not fname.startswith("output"):
                    file_path = os.path.join(root, fname)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        out_file.write(f"{file_path}:\n{content}\n\n")
                    except Exception as e:
                        out_file.write(f"{file_path}：read fail, error message：{e}\n")

if __name__ == "__main__":
    folder = "./logs/2025_05_19_23_06_30"
    gather_logs(folder)
