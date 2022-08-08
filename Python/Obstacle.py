from BoundingBox import BoundingBox as bbox 

class Obstacle:
    id = 0
    
    def __init__(self, x0, y0, x1, y1, frame):
        self.id = self.id + 1
        boundingBox = bbox(x0, y0, x1, y1)
        positionHistory = []
        totalFrames = 1
        Timestamp = frame

    def isAlive():
        return 0
    def update():
        return 0
    def distance():
        return 0

def main():
    newObs = Obstacle(1, 1, 3, 3, 1)

    print(newObs)

if __name__ == "__main__":
    main()


