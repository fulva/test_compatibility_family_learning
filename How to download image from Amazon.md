### How to download image from Amazon

#### Generate image ID list 

Follow the step 1 to 3 of [Amazon Co-purchase Experiments](https://github.com/appier/compatibility-family-learning#amazon-co-purchase-experiments) to generate `all_id_pairs.txt`.

#### Create workspace

> This folder will be used to store images

```bash
mkdir -p ~/compatibility-family-learning/data/dyadic/original_images
cd ~/compatibility-family-learning/data/dyadic/original_images
```

#### Prepare downloader

Create a new python program, e.g. `fetch.py` with following contents:

```python
#!/usr/bin/env python

import os
import subprocess
import sys

with open(sys.argv[1], 'r') as f:

	for line in f.readlines():
		url = line.split('\n')[0].split()[-1]
		filename = url.split('/')[-1]

		if not os.path.isfile(filename):
			cmd = "curl -o {} {}".format(filename, url)
			try:
				subprocess.call(cmd, shell=True)
			except Exception:
				print("error when fetch: {}".format(url))
```

In this program:

  1. Get ID list from the first argument
  2. Get image url by split each line of the ID list
  3. Get filename from the url
  4. Download image via `curl` with image url


#### Download images

Download images via downloader (`fetch.py`) with the id list:

```bash
cd ~/compatibility-family-learning/data/dyadic/original_images
python fetch.py ../all_id_pairs.txt
```