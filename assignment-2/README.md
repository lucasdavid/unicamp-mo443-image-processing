# MO-443 Assignment-2

Name: Lucas David
RA: 188972
Email: <lucas.david@ic.unicamp.br>


### File Descriptions

| File                   | Description                                  |
| ---------------------- | -------------------------------------------- |
| code/transform.py      | Util for applying the FFT to images          |
| exploration.ipynb      | Exploration work performed during the work   |
| report.pdf             | Written report                               |


### Usage

```bash
pip install -r code/requirements.txt

python code/transform.py --help
python code/transform.py -i /path/to/image.png -o /path/to/image-results/
python code/transform.py -i /path/to/image.png -o /path/to/image-results/ \
       --low-pass 10           \
       --high-pass 30          \
       --compression-rate 99.9 \
       --rotation 70
```
