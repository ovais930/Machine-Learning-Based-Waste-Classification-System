# PCA9685 MicroPython Driver

import time
from machine import I2C

_MODE1 = 0x00
_PRESCALE = 0xFE

class PCA9685:
    def __init__(self, i2c: I2C, address: int = 0x40):
        self.i2c = i2c
        self.address = address
        self.reset()

    def reset(self):
        self.write(_MODE1, 0x00)

    def write(self, reg, value):
        self.i2c.writeto_mem(self.address, reg, bytes([value]))

    def read(self, reg):
        return self.i2c.readfrom_mem(self.address, reg, 1)[0]

    def freq(self, freq):
        prescale = int(25000000.0 / (4096 * freq) - 1)
        oldmode = self.read(_MODE1)
        newmode = (oldmode & 0x7F) | 0x10
        self.write(_MODE1, newmode)
        self.write(_PRESCALE, prescale)
        self.write(_MODE1, oldmode)
        time.sleep_ms(5)
        self.write(_MODE1, oldmode | 0xa1)

    def pwm(self, channel, on, off):
        reg = 0x06 + 4 * channel
        self.write(reg, on & 0xFF)
        self.write(reg + 1, on >> 8)
        self.write(reg + 2, off & 0xFF)
        self.write(reg + 3, off >> 8)

    def duty_u16(self, channel, duty):
        duty = min(65535, max(0, duty))
        off = duty >> 4
        self.pwm(channel, 0, off)