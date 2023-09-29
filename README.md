# LightningFlow
A project template using Pytorch-Lightning. Thanks to the project of [Pytorch-Lightning-Template](https://github.com/miracleyoo/pytorch-lightning-template/tree/master.)



## File Structure

```
root-
	|-data
		|-__init__.py
		|-data_interface.py
		|-CLS_dataset.py ==> CLS_Dataset
                |-XXX_dataset.py ==> XXX_Dataset
		|-...
	|-model
		|-__init__.py
		|-model_interface.py
		|-CLS_model.py ==> CLS_Model
                |-YYY_model.py ==> YYY_Model
		|-...
	|-main.py
```

Suggestions:
- It is essential to maintain consistency between the names of XXX_dataset.py and XXX_Dataset in XXX, as well as YYY_model.py and YYY_Model in YYY. The prefix would be used in the "instancialize()" function of "data_interface.py" and "model_interface.py", respectively.
- To initiate your own project, it is necessary to rewrite "XXX_dataset.py" and "YYY_model.py" according to your specific requirements. However, please make an effort to keep "data_interface.py" and "model_interface.py" unchanged as much as possible to maintain the stability and compatibility of the existing codebase.
- I recommand you to use the OneCycle scheduler + AdamW optimizer.
- Please believe me, this framework will greatly enhance your coding efficiency.