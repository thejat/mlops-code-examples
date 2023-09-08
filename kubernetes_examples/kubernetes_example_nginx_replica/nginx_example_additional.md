## Before running kubectl apply:

 - Change to minikube's docker thats on the master using
        minikube docker-env
        eval $(minikube -p minikube docker-env)
 - Build the image for this docker runtime (go into the docker_example folder and then) using
        docker build -t minikube_weather .

## Running kubectl apply:
	
	kubectl apply -f weather_pod_example.yaml

## After running kubectl apply:

 - Expose the container to the world
		kubectl expose pod test --type=NodePort --port=5000

 - Make an example request (find the cluster's IP and port as in the imperative example)
 		curl http://192.168.99.101:30325?msg=Chicago