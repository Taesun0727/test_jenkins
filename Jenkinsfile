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
                script {
                    def existingContainer = sh(script: 'docker ps -a | grep flask-container', returnStdout: true).trim()
                    if (existingContainer) {
                        sh "docker stop $existingContainer || true"
                        sh "docker rm $existingContainer || true"
                    }
                }
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 9999:9999 --name flask-container myflaskapp'
            }
        }
    }
}