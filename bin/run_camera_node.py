#!/usr/bin/env python3
# Executable script for starting a camera node webserver

import asyncio

from webapp.main import main

if __name__ == '__main__':
    asyncio.run(main())
