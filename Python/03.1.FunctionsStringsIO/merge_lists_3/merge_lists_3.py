import typing as tp
import heapq


def merge(input_streams: tp.Sequence[tp.IO[bytes]], output_stream: tp.IO[bytes]) -> None:
    """
    Merge input_streams in output_stream
    :param input_streams: list of input streams. Contains byte-strings separated by "\n". Nonempty stream ends with "\n"
    :param output_stream: output stream. Contains byte-strings separated by "\n". Nonempty stream ends with "\n"
    :return: None
    """
    heap: tp.List[int] = []
    for stream in input_streams:
        while byte := stream.readline():
            elem = int(byte.decode('utf-8').rstrip('\n'))
            heapq.heappush(heap, elem)
    while heap:

        value = bytes(str(heapq.heappop(heap)), encoding='utf-8')
        output_stream.write(value + b'\n')
