# Segmentation models

### Run tests
```bash
$ docker build -f docker/Dockerfile.dev -t smp:dev .
$ docker run --rm -v $(pwd):/tmp/smp smp:dev pytest
```