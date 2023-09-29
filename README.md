# LightningFlow
A project template using Pytorch-Lightning. Thanks to the project of [Pytorch-Lightning-Template](https://github.com/miracleyoo/pytorch-lightning-template/tree/master.)



## File Structure

```
root-
	|-data
		|-__init__.py
		|-data_interface.py
		|-CLS_dataset1.py ==> CLSDataset
        |-XXX_dataset2.py ==> XXXDataset
		|-...
	|-model
		|-__init__.py
		|-model_interface.py
		|-CLS_model1.py ==> CLSModel
        |-YYY_model2.py ==> YYYModel
		|-...
	|-main.py
```

Suggestions:
- It is essential to maintain consistency between the names of XXX_dataset.py and XXXDataset in XXX, as well as YYY_model.py and YYY_model in YYY. The prefix would be used in the "instancialize()" function of "data_interface.py" and "model_interface.pt", respectively.
- To initiate your own project, it is necessary to rewrite "XXX_dataset.py" and "YYY_model.py" according to your specific requirements. However, please make an effort to keep "data_interface.py" and "model_interface.py" unchanged as much as possible to maintain the stability and compatibility of the existing codebase.