pipeline {
    agent any

    stages {
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t myflaskapp AI/'
            }
        }

        stage('Stop and Remove Existing Container') {
            steps {
                sh 'docker stop myflaskapp-container || true'
                sh 'docker rm myflaskapp-container || true'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 9999:9999 --name myflaskapp-container myflaskapp'
            }
        }
    }
}