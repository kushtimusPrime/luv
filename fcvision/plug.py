import kasa
import asyncio
import threading


class Plug:
    def __init__(self,IP='10.0.10.123'):
        self.plug=kasa.SmartPlug(IP)
        self.loop=asyncio.get_event_loop()
    def turn_on(self):
        self.loop.run_until_complete(self.plug.turn_on())
    def turn_off(self):
        self.loop.run_until_complete(self.plug.turn_off())