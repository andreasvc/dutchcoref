# https://pythonspeed.com/articles/base-image-python-docker-images/
FROM python:3.9-bullseye as base
MAINTAINER Andreas van Cranenburgh <a.w.van.cranenburgh@rug.nl>
# https://snyk.io/blog/best-practices-containerizing-python-docker/
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
ENV HOME=/usr

# FIXME: model name should go in a configuration file
RUN python -c 'from transformers import AutoTokenizer, AutoModel; \
	name="GroNLP/bert-base-dutch-cased"; \
	AutoTokenizer.from_pretrained(name); \
	AutoModel.from_pretrained(name)'

RUN groupadd -g 999 user && \
    useradd -r -u 999 -g user user
RUN mkdir --parents /usr/app/data /usr/app/templates /usr/groref && \
	chown --recursive user:user /usr/app /usr/.cache

WORKDIR /usr/groref
RUN wget https://bitbucket.org/robvanderg/groref/raw/cb1eb35a4955cc8adc7036eb4cf4ec57e1ccb392/ngdata

WORKDIR /usr/app
COPY --chown=user:user mentionspanclassif.* mentionfeatclassif.* pronounmodel.* ./
COPY --chown=user:user data data/
COPY --chown=user:user templates templates/
COPY --chown=user:user *.py ./

USER 999
RUN python -c 'import coref; coref.readngdata()'
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8899 --workers=2 --preload"
CMD ["gunicorn", "web:APP"]
