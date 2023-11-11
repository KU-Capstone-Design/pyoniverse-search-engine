"""
[KoSpacing의 속도 측정]
이유: 현재 모듈에서 Spacing을 매번 처리하는데, 10,000개 정도의 데이터를 처리하는데 5분정도 걸린다.
테스트 내용: 1000개의 무작위 데이터로 연산을 10회 실시해 평균 속도를 구한다.
테스트 실행: kernprof -l -v -u 1e-3 profiles/time/profile_kospacing.py > profiles/time/report/profile_kospacing.txt(밀리초)
테스트 결과: Space.__call__ 자체가 시간이 오래걸린다(26sec/name)
결론: Spacing 결과를 저장하자
"""
import json
from random import sample

from line_profiler import profile
from pykospacing import Spacing


@profile
def execute_kospace(data: list):
    """
    Spacing 연산 수행
    :param data: products 데이터
    """
    spacing = Spacing()
    result = [spacing(d["name"], ignore="none") for d in data]


def profile_kospace(count: int = 10):
    """
    @step
    1. profiles/resource/products.json 가져오기
    2. 랜덤 100개 선택 x count
    3. execute_kospace(data) 실행
    """
    with open("profiles/resource/products.json", "r") as fd:
        data = json.load(fd)
        for _ in range(count):
            samples = sample(data, k=1000)
            execute_kospace(samples)


if __name__ == "__main__":
    profile_kospace()
