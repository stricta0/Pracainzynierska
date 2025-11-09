def create_line(data_base, hidden_size, model_name, log_file_name=None, log_dir=None):
    if log_dir is None and log_file_name is not None:
        log_dir = log_file_name
    if log_dir is not None and log_file_name is None:
        log_file_name = log_dir
    if log_file_name is None and log_dir is None:
        log_file_name = create_log_fi_names(data_base, model_name, hidden_size)
        log_dir = create_log_fi_names(data_base, model_name, hidden_size)
    if model_name == "MLP":
        return f"python {get_python_program(model_name)} --dataset {data_base} --hidden {hidden_size} --batch_size 256 --epochs 5000 --patience 500 --lr 1e-3 --momentum 0.9 --seed 42 --log_file {log_file_name} --model_name {model_name} --log_dir wyniki/{log_dir}\n"
    if model_name == "RFF":
        return f"python {get_python_program(model_name)} --dataset {data_base} --n_features {hidden_size} --batch_size 256 --epochs 5000 --patience 500 --lr 1e-3 --momentum 0.9 --seed 42 --log_file {log_file_name} --model_name {model_name} --log_dir wyniki/{log_dir}\n"
    if model_name == "CNN":
        return f"python {get_python_program(model_name)} --dataset {data_base} --ch1 {hidden_size} --ch2 {hidden_size*2} --batch_size 256 --epochs 5000 --patience 500 --lr 1e-3 --momentum 0.9 --seed 42 --log_file {log_file_name} --model_name {model_name} --log_dir wyniki/{log_dir}\n"

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
n_features = [
    500, 1000, 2000, 4000, 5000,
    7500, 10000, 12_5000, 15_000, 17_500, 20_000,
    30_000, 40_000, 50_000, 100_000
]
chi_no = [4, 8, 12, 16, 32, 50, 64, 80, 100, 110, 128,
          150, 175, 200, 256, 300, 400, 512, 1024, 2048]


models_names = ["MLP", "RFF", "CNN"]

def create_all_double_descent(data_bases, models_names):
    all = ""
    for model_name in models_names:
        for data_base in data_bases:
            locla_tab = None
            if model_name == "CNN":
                locla_tab = chi_no
            if model_name == "MLP":
                locla_tab = hidden
            if model_name == "RFF":
                locla_tab = n_features
            for hidden_no in locla_tab:
                all +=  create_line(data_base, hidden_no, model_name)
    return all

print(create_all_double_descent(data_bases, ["MLP"]))
