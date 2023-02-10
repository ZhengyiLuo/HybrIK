import asyncio
from random import randrange


async def main_function():
    n = randrange(2)
    if n:
        return 1
    else:
        return None


def backup_function():
    return 1


async def run():
    jobs1 = [main_function() for _ in range(0, 5)]
    jobs2 = [backup_function() for _ in range(0, 5)]

    _res = await asyncio.gather(*jobs1)
    if len(_res) < 5:
        _res = await asyncio.gather(*jobs2)


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(run())