import boto3
import requests
import time
import json

# SageMaker 엔드포인트 이름
ENDPOINT_NAME = "team3-endpoint2"  # 실제 SageMaker 엔드포인트 이름 입력
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event:", json.dumps(event, indent=2))
    try:
        # S3 이벤트에서 버킷 이름과 파일 키 추출
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        file_name = file_key.split('/')[-1]

        # diecastUuid 추출: 파일명에서 추출 (예: frame_3_2_001.jpg → diecastUuid = "3")
        try:
            diecastUuid = file_name.split('_')[1]  # 파일명에서 두 번째 '_' 뒤의 숫자 추출
        except IndexError:
            diecastUuid = "default"  # 파일명에 diecastUuid가 없으면 기본값 설정
            print("diecastUuid not found in file name. Using default.")

        # photoPosition 추출: 파일명에서 추출 (예: frame_3_2_001.jpg → photoPosition = "2")
        try:
            photoPosition = file_name.split('_')[2]  # 파일명에서 세 번째 '_' 뒤의 숫자 추출
        except IndexError:
            photoPosition = "default"  # 파일명에 photoPosition이 없으면 기본값 설정
            print("photoPosition not found in file name. Using default.")

        print(f"Processing file: {file_key} from bucket: {bucket_name}, diecastUuid: {diecastUuid}, photoPosition: {photoPosition}")
        
        # 숫자 추출 (예: frame_3_2_001.jpg → "001")
        try:
            number_str = file_name.split('_')[-1].split('.')[0]
            number = int(number_str)  # 정수로 변환
            
            # 파일 이름이 5의 배수 + 1로 끝나는 경우
            if (number - 1) % 5 == 0:
                patch_url = f"http://18.205.110.55:8080/diecast/save/1"
                print("Starting HTTP PATCH request...")
                patch_response = requests.post(patch_url, timeout=10)

                print(f"PATCH POST completed: {patch_response.status_code}, {patch_response.text}")
        except Exception as e:
            print(f"Error in PATCH logic: {str(e)}")

        # S3에서 파일 다운로드
        start_time = time.time()
        print(f"Starting file download from S3: {file_key}")
        s3 = boto3.client('s3')
        local_file_path = f"/tmp/{file_key.split('/')[-1]}"  # Lambda의 임시 경로
        s3.download_file(bucket_name, file_key, local_file_path)
        print(f"File download completed: {local_file_path}")
        
        # SageMaker 엔드포인트로 이미지 추론 요청
        print(f"Invoking SageMaker endpoint: {ENDPOINT_NAME}")
        with open(local_file_path, 'rb') as image_file:
            payload = image_file.read()  # 이미지 파일 바이너리 데이터 읽기
            
            # SageMaker 엔드포인트 호출
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/x-image',  # 이미지 데이터 타입 설정
                Body=payload
            )
        
        # SageMaker 추론 결과 확인
        print("SageMaker inference response received")
        result = json.loads(response['Body'].read().decode('utf-8'))
        print(f"Inference result: {result}")
        
        # 추론 결과에서 OK와 NG 판별 (값 반전)
        if "predicted_class" in result:
            photo_ng_type = 1 if result["predicted_class"] == 0 else 0  # 0 → 1, 1 → 0
        else:
            raise ValueError("Unexpected inference result format")
        
        # 첫 번째 POST 요청: 개별 사진 정보 전송
        post_url = f"http://18.205.110.55:8080/diecast/{diecastUuid}"
        print("Starting HTTP POST request for individual photo...")
        with open(local_file_path, 'rb') as file:
            files = {
                "photoFile": (file_name, file, "image/jpeg")  # 파일 이름 및 MIME 타입 설정
            }
            data = {
                "photoPosition": photoPosition,  # 추출한 photoPosition 값 사용
                "photoNgtype": str(photo_ng_type),  # 추론 결과에 따라 0 또는 1 설정
                "photoCroplt": "1.1",
                "photoCroprb": "8.8"
            }
            response = requests.post(post_url, data=data, files=files, timeout=10)
        
        print(f"Individual photo POST completed: {response.status_code}, {response.text}")
        
        # 두 번째 POST 요청: 파일 이름이 0 또는 5로 끝나는 경우 추가 전송
        if number % 5 == 0:
            save_url = f"http://18.205.110.55:8080/diecast/patch/{diecastUuid}"
            print("Starting HTTP POST request for save endpoint...")
            save_response = requests.patch(save_url, timeout=10)
            print(f"Save POST completed: {save_response.status_code}, {save_response.text}")
        
        return {
            "statusCode": 200,
            "body": f"Processed file: {file_name}"
        }
    
    except Exception as e:
        print(f"Error processing event: {str(e)}")
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
