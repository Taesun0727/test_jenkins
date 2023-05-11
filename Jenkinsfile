pipeline {
    agent any

    stages {
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t myflaskapp AI/'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 9999:9999 myflaskapp'
            }
        }
    }
}