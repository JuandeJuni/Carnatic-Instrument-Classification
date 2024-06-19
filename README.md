# Carnatic-Instrument-Classification

To compile and run use docker:

```bash
docker build -t mtl-app:latest .
docker run -p 8080:8080 mtl-app:latest
```
After this, you can access the web app through localhost:8080, and you will find a default precomputed sample

The web app is also deployed on gcloud. It can be accessed through https://carnaticic-upf.web.app/, however, some functionalities are not available.

## Functionalities

- You can click the graph of intervals and an interactive graph will open on a new tab
- You can download the generated graphs
- You can upload a file and make it predict and generate new graphs (only works locally)