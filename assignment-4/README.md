# MO-443 Assignment-4

Name: Lucas David
RA: 188972
Email: <lucas.david@ic.unicamp.br>


### File Descriptions

| File                   | Description                                |
| ---------------------- | -------------------------------------------|
| code/texture.py        | Util for analyzing texture in image files  |
| exploration.ipynb      | Exploration work performed during the work |
| report.pdf             | Written report                             |


### Usage

```bash
pip install -r requirements.txt

python texture.py -h
python texture.py -i /path/to/textures/ -o /path/to/texture-results/

python texture.py -i /path/to/textures/ \
                  -o /path/to/texture-results/ \
                  --distance kld
                  --glcm-props dissimilarity,correlation
```
