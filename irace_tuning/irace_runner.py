#!/home/callumm/.virtualenvs/enc-vpn-uncertainty-class-repl/bin/python
from runs.run_calibrate import calibrate_and_test


# used 'dos2unix' so that I can run this with wsl irace
from parameter_store import ParameterStore
def main():
    import sys
    from runs.run_train import init_dataloaders, train_pnet
    import random
    import numpy as np
    import torch
    import logging

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG,\
                        datefmt='%Y-%m-%d %H:%M:%S',\
                        filename='irace_runner.log', filemode='a')
    logger = logging.getLogger(__name__)


    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    # get random run split to use:
    run = np.random.randint(0, 10)
    logger.debug(
        f"Configuration ID: {configuration_id}, Instance ID: {instance_id}, Seed: {seed}, Instance: {instance}, Data split: run{run}")


    # set seeds
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # data parameters
    n_classes = 7
    n_way = 7

    #default values
    lr = 0.03421
    weight_decay = 0.002298
    hidden_size = 256
    output_size = 64
    dropout = 0.107547
    hidden_layers = 3
    while cand_params:
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        logger.debug(f"Parameter parsed: {param} = {value}")
        match param:
            case "--lr":
                lr = float(value)
            case "--weight_decay":
                weight_decay = float(value)
            case "--hidden_size":
                hidden_size = int(value)
            case "--dropout":
                dropout = float(value)
            case "--hidden_layers":
                hidden_layers = int(value)
    # vpn data
    #data_path = "../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/min_max_normalized/run0/frac_80"
    data_path = f"../../forest_ct_data/covertype_data/with_wavelets/stable_cal_fraction/min_max_normalized/run0/frac_80" # forest data
    train_dl, val_dl = init_dataloaders(full_path=data_path, n_classes=n_classes, n_way=n_way)

    params = ParameterStore(run_type="irace_with_wavelets", n_classes=7, x_dim=62, n_way=7,
                                              dropout=dropout,
                                              lr=lr, weight_decay=weight_decay, hidden_dim=hidden_size, hidden_layers=hidden_layers)

    acc, loss, pnet= train_pnet(train_dl, val_dl, params)

    logger.info(f"Training completed. Train Accuracy: {acc}, Train Loss: {loss}")
    print(loss)



if __name__ == '__main__':
    main()
