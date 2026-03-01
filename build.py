"""
Build and deployment script for Table Analysis Service
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run command and print status"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✅ Success: {description}")
    return True


def build_docker_image():
    """Build Docker image"""
    return run_command(
        "docker build -t financial-table-analysis:latest .",
        "Building Docker image"
    )


def start_services():
    """Start docker-compose services"""
    return run_command(
        "docker-compose up -d",
        "Starting docker-compose services"
    )


def stop_services():
    """Stop docker-compose services"""
    return run_command(
        "docker-compose down",
        "Stopping docker-compose services"
    )


def check_health():
    """Check service health"""
    import requests
    import json
    
    print("\n" + "="*60)
    print("  Checking Service Health")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        health = response.json()
        
        print(f"Service Status: {health['status']}")
        print(f"  SAM3: {'✅' if health['sam3_ready'] else '❌'}")
        print(f"  Ollama: {'✅' if health['ollama_ready'] else '❌'}")
        print(f"  Models Dir: {'✅' if health['models_dir_exists'] else '❌'}")
        
        return health['status'] == 'healthy'
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("  Make sure services are running: docker-compose up")
        return False


def push_to_ecr(image_uri):
    """Push Docker image to ECR"""
    commands = [
        f"aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin {image_uri.rsplit('/', 1)[0]}",
        f"docker tag financial-table-analysis:latest {image_uri}",
        f"docker push {image_uri}"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"ECR: {cmd[:40]}..."):
            return False
    return True


def show_usage():
    """Show build script usage"""
    print("""
Usage: python build.py <command>

Commands:
  build              Build Docker image locally
  start              Start services with docker-compose
  stop               Stop docker-compose services
  health             Check service health
  ecr <image_uri>    Push image to AWS ECR
  dev                Build and start for local development
  clean              Stop and remove all containers/volumes

Examples:
  python build.py build
  python build.py start
  python build.py health
  python build.py ecr 123456789.dkr.ecr.us-east-1.amazonaws.com/financial-table-analysis:latest
  python build.py dev
""")


def main():
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "build":
        build_docker_image()
    
    elif command == "start":
        start_services()
        print("\nWaiting for services to be ready...")
        import time
        time.sleep(10)
        check_health()
    
    elif command == "stop":
        stop_services()
    
    elif command == "health":
        check_health()
    
    elif command == "ecr":
        if len(sys.argv) < 3:
            print("Error: ECR image URI required")
            print("Usage: python build.py ecr <image_uri>")
            return
        push_to_ecr(sys.argv[2])
    
    elif command == "dev":
        print("🚀 Starting development environment...")
        if not build_docker_image():
            return
        if not start_services():
            return
        print("\n⏳ Waiting for services to be ready (30 seconds)...")
        import time
        time.sleep(30)
        check_health()
        print("\n✅ Development environment ready!")
        print("📖 API Docs: http://localhost:8080/docs")
        print("🔍 Health: http://localhost:8080/health")
    
    elif command == "clean":
        run_command("docker-compose down -v", "Cleaning up containers and volumes")
    
    else:
        print(f"Unknown command: {command}")
        show_usage()


if __name__ == "__main__":
    main()
