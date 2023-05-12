pipeline {
    agent any

    stages {
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t myfastapiapp AI/'
            }
        }

        stage('Stop and Remove Existing Container') {
            steps {
                script {
                    def existingContainer = sh(script: 'docker ps -a --filter "name=fastapi" --format "{{.ID}}"', returnStdout: true).trim()
                    if (existingContainer) {
                        sh "docker stop $existingContainer || true"
                        sh "docker rm $existingContainer || true"
                    }
                }
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 8700:8700 --name fastapi myfastapiapp'
            }
        }
    }
}