first run the scraper you will get folders in `data/img`
copy the folders to `images` folder
then run `dataset maker.py` (dont forget to make `other` folder if it not in `dataset` folder also add your nocaptchaai key)
then run `train.py`
it will give you {'animal with four legs': 0, 'other': 1}
make list of the somethings in same order (['animal with four legs','other'])
then replace it in `testing.py` line `13`
then put test images in `testing` folder
then run `testing.py` to see the results
