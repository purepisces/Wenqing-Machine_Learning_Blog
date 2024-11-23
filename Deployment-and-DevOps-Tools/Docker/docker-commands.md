```
docker ps -a
```
```css
CONTAINER ID   IMAGE                          COMMAND                  CREATED          STATUS                        PORTS                    NAMES
52939179746f   recommendation-system:latest   "python app.py"          14 minutes ago   Exited (255) 13 seconds ago   0.0.0.0:8001->5000/tcp   recommendation-system
e41f3f4f7c49   seem-backend:latest            "flask run --host=0.…"   17 minutes ago   Exited (255) 13 seconds ago   0.0.0.0:8000->5000/tcp   flask-app
19728bae9607   mongo:latest                   "docker-entrypoint.s…"   9 days ago       Exited (0) 26 seconds ago                              mongo-container
(base) 
```
To delete all containers (whether running or exited):
```
docker rm $(docker ps -aq)
```
