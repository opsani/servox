# Example Application Templates for Servox

The documents herein are provided as a sandbox to allow (potential) customers to begin experimenting with servo quickly and simply.
They are runnable as is but mostly intended to serve as a starting point for individualized POCs.

## Kubernetes

Requirements: minikube installed

For mainline optimizations of deployments pinned at a single replica, NodePort services are sufficient for networking into the kubernetes environment.
For canary optimizations and other cases where multiple instances of the app are being orchestrated, it becomes neccessary to introduce a load balancer
to accurately replicate real world networking conditions. 

Mainline optimization examples implement single replicas with NodePorts for simplicity but the load balancer section of the canary instructions can also
be applied to support mainline deployments with more than one replicas

### Mainline Optimization

1. Run `minikube start`
1. Run `kubectl apply -f prometheus.yaml`
1. Run `kubectl apply -f web-svc-np.yaml`
1. Run `kubectl apply -f web-dep.yaml` (skip this step if deploying ArgoCD rollout)
    1. **For Argo CD Rollouts ONLY**: Run `kubectl create namespace argo-rollouts`
    1. Run `kubectl apply -n argo-rollouts -f https://raw.githubusercontent.com/argoproj/argo-rollouts/stable/manifests/install.yaml`
    1. Run `kubectl apply -f web-rol.yaml`
1. Copy the `--- # Canary deployment optmization` or `--- # Canary rollout optmization` section of the `example_app/example_servo.yaml` to the `servo.yaml` in the root or your project folder
1. Run `minikube service -n opsani-monitoring prometheus`; copy the URL column to replace the `base_url` of the config section for `prometheus` in `servo.yaml`
1. Run `minikube service web`; copy the URL column to replace the `target` of the config section for `vegeta` in `servo.yaml`
1. Run `servo check` to verify connectivity
1. Run `servo run` to start an optimization



### Canary Optimization

1. Run `minikube start`
1. Run `minikube addons enable metallb`
1. Run `minikube addons configure metallb`
1. Determine and enter `Load Balancer Start/End IP`s based on the configuration of your LAN's DHCP (use an address range in the same network as your host address but 
one that will not be assigned to your wifi users)
1. Run `kubectl describe configmap config -n metallb-system` and validate the following section format:

    ```yaml
    - name: default
    protocol: layer2
    addresses:
    - 192.168.1.45-192.168.1.60
    ```
1. **NOTE** In some cases, the config will not take effect and you will instead see a blank range for addresses (eg. `  - -`). In such cases, proceed with the following sub list  
    1. Run `export KUBE_EDITOR="nano"`
    1. Run `kubectl edit configmap config -n metallb-system`
    1. Update the `addresses` yaml with your desired range
    1. Press CTRL + x to exit and cofirm the prompt to save changes

1. Run `kubectl apply -f prometheus.yaml`
1. Run `kubectl apply -f web-svc-lb.yaml`
1. Run `kubectl apply -f web-dep.yaml` (skip if deploying ArgoCD rollout)
    1. **For Argo CD Rollouts ONLY**: Run `kubectl create namespace argo-rollouts`
    1. Run `kubectl apply -n argo-rollouts -f https://raw.githubusercontent.com/argoproj/argo-rollouts/stable/manifests/install.yaml`
    1. Run `kubectl apply -f web-rol.yaml`
1. Copy the `--- # Canary deployment optmization` or `--- # Canary rollout optmization` section of the `example_app/example_servo.yaml` to the `servo.yaml` in the root or your project folder
1. Run `minikube service -n opsani-monitoring prometheus`; copy the URL column to replace the `base_url` of the config section for `prometheus` in `servo.yaml`
1. Run `minikube service web`; copy the URL column to replace the `target` of the config section for `vegeta` in `servo.yaml`
1. Run `servo check` to verify connectivity
1. Run `servo run` to start an optimization


### Cleanup

1. Run `minikube delete`
