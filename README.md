# BS_net
Code for Physics-Informed B-Spline Nets



Number of control points abaltion experiments: run `num_ctrl_pts.m`.



Results for 3D Fokker-Plank equations, run

```bash
python fplanck/new_code/harmonic_3d.py
python fplanck/new_code/tilted_bigaussia_3d.py
```



For BS net implementation, see [BS_Net.ipynb](https://colab.research.google.com/drive/1yDTg6OgvhlIR4NYX2GRo8fHKlpNKmNsp#scrollTo=SgbyRz4aAjIQ) on Google Colab.



A demo for the BS net for 3D Fokker-Planck functions can be found on Colab at [BS_Net_3D_Planck.ipynb](https://colab.research.google.com/drive/10cTzX1pLu9-7-7sOxkfTRrinFEOUKv4E#scrollTo=yzS2ntTDqRsx)


Simulation code for BS net and other baselines on the recovery probability problem can be found at

[ICLR25_BS_Net.ipynb](https://colab.research.google.com/drive/1MyzMYOMIKd_rSPKm7aUDhIOc0tHpRY_c)

[ICLR25_PINN_Varying_a.ipynb](https://colab.research.google.com/drive/1etEQyCTbO7sSD-lBFmc_fLmDz2n93iMl)

[ICLR25_DeepONet_Varying_a.ipynb](https://colab.research.google.com/drive/1q68svEhPPuMC1PnQeapEckj7xuDzsaIk)

The data generation for Burgers' equations is at: [Burgers_Data_Generation.ipynb](https://colab.research.google.com/drive/15rg3-UzisiK1HLo8t43F4lnveLME-Zon)

BS Net for Burgers' equations is at: [Burgers_BS_Net.ipynb](https://colab.research.google.com/drive/1kVaBsBpciPP2T2FofcddTQoX68p7gZrk)

BS Net for Advection equation is at [Advection_Beta_Phase.ipynb](https://colab.research.google.com/drive/10mRqRjwkjXfHYuW7g2AdMsXhU4naAHbB#scrollTo=j7ZIinAb44Qa)

PINN results for he 3D heat equation is at: [Heat_3D_PINN_ICLR_setting.ipynb](https://colab.research.google.com/drive/17SzFg9x7dhsP7NOPdW1h8KjPuMugwXpl)


Comparison plots in the paper can be generated via: [PINN_varying_comparison_with_BS.ipynb](https://colab.research.google.com/drive/1a1Y4d4qCYto5UVzwv_waG3wSYMpOwCKA)


BS Net for diffusion equations on the trapezoid domain can be found at: Square training data [BS_Net_Trapezoid.ipynb](https://colab.research.google.com/drive/1g9FITHacs19kVaoY2qzsqoGx2G1bE8yi), Mapped trapezoid training data [BS_Net_Trapezoid_Mapped_Training_Data.ipynb](https://colab.research.google.com/drive/1Qku5ZQqQq5Ra1-RFmRrBfuEkW3iaRoXn#scrollTo=V02xYzE6p542)


Ablation experiment code is under folder 'Neurips_25', with conda environment file 'pi-dbsn-env.yml'.

Loss curves, trade-off plots are generated via [ICLR25_Ablation_Visualization.ipynb](https://colab.research.google.com/drive/1GcWFMPrMgP5fcEEP3O9BNY8qFtV67Asa#scrollTo=_mETEPHNbV8q)

Gradient and Hessian comparison is at [BS_Net_Grad/Hessian.ipynb](https://colab.research.google.com/drive/1iRflZMytZ4Kq5QQ5KyKeUUXznClJaExN#scrollTo=pkaQqt1Wj94J)

Generalization error experiment on advection equations is at [Advection_Generalization_Error.ipynb](https://colab.research.google.com/drive/1sak9dwM5PnCs7cHr623uqULVT3oFDTvL#scrollTo=KPPiBHIMyPZF)


### REVISED

To run the code for BS net

```bash
pip install -e .
```

#### Training 
creates checkpoints
```bash
python scripts/cli.py  fit -c scripts/config_fp.yaml
```

now with a config file like this:
```
trainer:
  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: "experiments/{name}/{run_name}/logs"
```

this will make a checkpoint model here (for example - where RecurrentModel is the name of the model):
```
experiment/RecurrentModel/20241117-225850/checkpoints
```


#### Testing 
creates output pytorch files

```bash
python scripts/cli.py  test -c scripts/config_fp.yaml
```
this will create a files in (for example):

```
experiment/RecurrentModel/20241117-225850/test_results
```


#### Creating plots 

makes pdf files (assuming you have two files training and testing)
```bash
python plot.py training.pt testing.pt
```
