# Code File Descriptions
| File              | Description                                                                                                                                                                     |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset.py        | Definition of the Dataset class for training and evaluation, with dataset paths also specified here.                                                                            |
| evaluator.py      | Code for evaluating the model during training.                                                                                                                                  |
| inference.py      | Code for inferring Super-Resolution (SR) results.                                                                                                                               |
| losses.py         | Definitions of the loss functions.                                                                                                                                              |
| test_metrics.py   | Code for testing Image Quality Assessment (IQA) metrics.                                                                                                                        |
| train_3_loss.py   | Implementation of MOBOSR with settings identical to ESRGAN, but employing multi-objective Bayesian optimization to dynamically adjust loss weights during the training process. |
| train_all_loss.py | Implementation of MOBOSR using all losses.                                                                                                                                      |
| train_origin.py   | Standard implementation of ESRGAN, also utilized during the pre-training phase of MOBOSR.                                                                                       |
| utils.py          | Various utilities.                                                                                                                                                              |
# Pre-trained Model Weights
| Model                            | Download Link                                                                                     |
|----------------------------------|---------------------------------------------------------------------------------------------------|
| Our-a                            | [Download](https://drive.google.com/file/d/1GnltJSNIJnvnO2CCyXOI1RZaEPKp0J7V/view?usp=drive_link) |
| Our-b                            | [Download](https://drive.google.com/file/d/1EGvElFHZ7-SHvyXCoO2GzjaSezE-M4QG/view?usp=drive_link) |
| Our-c                            | [Download](https://drive.google.com/file/d/1Kup-TcfaTOPSpFPf1lQPlkrNfeto_6Xm/view?usp=drive_link) |
| Generator at Pre-train phase     | [Download](https://drive.google.com/file/d/1yXnL-9NBYX7zRNIUaEDuCLRL_8XrhrZr/view?usp=drive_link) |
| Discriminator at Pre-train phase | [Download](https://drive.google.com/file/d/1nvAAemRYlHshQh7997EcnTIsxH06Of6C/view?usp=drive_link) |