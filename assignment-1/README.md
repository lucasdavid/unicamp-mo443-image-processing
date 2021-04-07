# MO-443 Assignment-1

Name: Lucas David
RA: 188972
Email: <lucas.david@ic.unicamp.br>


### File Descriptions

| File                   | Description                                  |
| ---------------------- | -------------------------------------------- |
| code/filters.py        | Util for applying the filters in image files |
| code/test_filters.py   | Test for the functions in `filters.py`       |
| exploration.ipynb      | Exploration work performed during the work   |
| report.pdf             | Written report                               |


### Usage

```bash
pip install -r requirements.txt

python filters.py -h
python filters.py -i /path/to/image.png -o /path/to/image-results/
```


#### Running Tests

```bash
pip install -r requirements-dev.txt

pytest
```
