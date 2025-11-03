def create_line(data_base, hidden_size, model_name, log_file_name=None, log_dir=None):
    if log_dir is None and log_file_name is not None:
        log_dir = log_file_name
    if log_dir is not None and log_file_name is None:
        log_file_name = log_dir
    if log_file_name is None and log_dir is None:
        log_file_name = create_log_fi_names(data_base, model_name, hidden_size)
        log_dir = create_log_fi_names(data_base, model_name, hidden_size)
    return f"python {get_python_program(model_name)} --dataset {data_base} --hidden {hidden_size} --batch_size 256 --epochs 5000 --patience 500 --lr 1e-3 --momentum 0.9 --seed 42 --log_file {log_file_name} --model_name {model_name} --log_dir wyniki/{log_dir}\n"

def get_python_program(model_name):
    if model_name == "MLP":
        return "mlp_przez_mymodel.py"
    elif model_name == "RFF":
        return "rff_przez_mymodel.py"
    elif model_name == "CNN":
        return "cnn_przez_mymodel.py"

def create_log_fi_names(data_base, model_name, hidden_size):
    return f"{model_name.lower()}_{data_base}_hidden_{hidden_size}.txt"

data_bases = ["mnist", "fashion", "emnist_balanced", "emnist_byclass"]
hidden = [
          20, 50, 75, 90, 100, 110, 130, 150, 175, 200, 250, 300,
          400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500,
          10_000, 20_000, 50_000, 100_000
        ]
models_names = ["MLP", "RFF", "CNN"]

def create_all_double_descent(data_bases, hidden, models_names):
    all = ""
    for model_name in models_names:
        for data_base in data_bases:
            for hidden_no in hidden:
                all +=  create_line(data_base, hidden_no, model_name)
    return all

print(create_all_double_descent(data_bases, hidden, ["RFF", "CNN"]))
