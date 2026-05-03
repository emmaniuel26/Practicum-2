"""
Jetson tegrastats-based energy logger.

This file:
- launches tegrastats
- parses power readings
- stores timestamped power samples
- integrates power over time into Joules
"""

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from config import POWER_KEYS, TEGRASTATS_INTERVAL_MS

# Common tegrastats power patterns seen on Jetson

POWER_PATTERNS = [
    re.compile(r"(VDD_IN|POM_5V_IN)\s+(\d+)mW/?(\d+)?mW?"),
    re.compile(r"(VDD_IN|POM_5V_IN)\s+(\d+)mW"),
]


@dataclass
class PowerSample:
    """
    One power reading at one timestamp.
    """
    timestamp: float
    power_watts: float


@dataclass
class TegraStatsLogger:
    """
    A simple logger that runs tegrastats in the background and records power samples.
    """
    interval_ms: int = TEGRASTATS_INTERVAL_MS
    process: Optional[subprocess.Popen] = None
    thread: Optional[threading.Thread] = None
    samples: list[PowerSample] = field(default_factory=list)
    _running: bool = False

    def _extract_power_watts(self, line: str) -> Optional[float]:
        """
        Parse a tegrastats output line and extract power in watts.
        """
        # Check each supported tegrastats power pattern
        
        for pattern in POWER_PATTERNS:
            matches = pattern.findall(line)
            if matches:
                for match in matches:
                    key = match[0]
                    
                    # Use only configured power rails
                    
                    if key in POWER_KEYS:
                        mw = float(match[1])
                        
                        # Convert milliwatts to watts
                        
                        return mw / 1000.0
        return None

    def _reader(self) -> None:
        """
        Background thread: read tegrastats line by line and store power samples.
        """
        assert self.process is not None

        # Continue reading while logger is active
        
        while self._running and self.process.stdout:
            line = self.process.stdout.readline()
            if not line:
                continue

            # Record timestamp for current power sample
            
            ts = time.time()

            # Extract power reading from tegrastats line
            
            power = self._extract_power_watts(line.strip())

            # Save valid power sample
            
            if power is not None:
                self.samples.append(PowerSample(timestamp=ts, power_watts=power))

    def start(self) -> None:
        """
        Start tegrastats logging.
        """
        # Clear old samples before starting a new run
        
        self.samples.clear()
        self._running = True

        # Launch tegrastats as subprocess
        
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        # Start background reader thread
        
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """
        Stop tegrastats logging.
        """
        self._running = False

        # Stop tegrastats subprocess
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()

        # Stop background reader thread
        
        if self.thread:
            self.thread.join(timeout=2)

    def energy_joules(self, start_ts: float, end_ts: float) -> float:
        """
        Integrate power over time between start_ts and end_ts.

        Uses trapezoidal integration:
        energy = integral(power * dt)
        """
        # Keep only samples within inference runtime window
        
        relevant = [s for s in self.samples if start_ts <= s.timestamp <= end_ts]

        # Need at least two points to calculate an area
        
        if len(relevant) < 2:
            return 0.0

        energy = 0.0

        # Integrate power over time using trapezoids
        
        for i in range(1, len(relevant)):
            dt = relevant[i].timestamp - relevant[i - 1].timestamp
            avg_power = (relevant[i].power_watts + relevant[i - 1].power_watts) / 2.0
            energy += avg_power * dt

        return energy
