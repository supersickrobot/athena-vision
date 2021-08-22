#!/usr/bin/env python3
# Executable script for starting a camera node webserver

import asyncio

from webapp.main import main

asyncio.run(main())
