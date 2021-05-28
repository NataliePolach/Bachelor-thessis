Skript pro trénování modelu- train.py
Skript s definicí architektury sítě - model.py
Skript pro transformace vstupů - transforms.py
Skript s loaders, acc - utils.py
Skript s definicí načtení datasetu - dataset.py
Skript pro zkošku funkčnosti natrénované sítě - model_test.py
Skript pro detekci nanočástic z naučeného modelu

/modely_natrenovane obsahují všechny trénované modely, model využitý pro konečnou detekci je ve složce 1 - unet_1.py
/train_image, /train_mask - obrazy pro trénovanání s příslušnými maskami
/val_img, val_mask - obrazy pro validaci sítě s příslušnými maskami
/detected_img - ukázka detekovaných nanočástic na validačním setu
