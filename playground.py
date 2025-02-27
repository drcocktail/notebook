import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import math

class GeometryNotebook:
    def __init__(self):
        self.points = {}  # Dictionary to store points {name: (x, y)}
        self.lines = []   # List to store lines [(point1_name, point2_name), ...]
        self.circles = [] # List to store circles [(center_name, point_on_circle_name), ...]
        self.reasoning_steps = []  # List to store reasoning steps
        self.construction_steps = []  # List to store construction steps
        
        # Setup the figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
    
    def add_point(self, name, x, y, source="direct"):
        """Add a point to the notebook."""
        if name in self.points:
            raise ValueError(f"Point {name} already exists.")
        
        self.points[name] = (x, y)
        construction = f"Created point {name} at coordinates ({x:.2f}, {y:.2f})"
        if source != "direct":
            construction += f" via {source}"
        self.construction_steps.append(construction)
        return name
    
    def add_line(self, point1_name, point2_name):
        """Add a line between two existing points using straightedge."""
        if point1_name not in self.points:
            raise ValueError(f"Point {point1_name} does not exist.")
        if point2_name not in self.points:
            raise ValueError(f"Point {point2_name} does not exist.")
        
        self.lines.append((point1_name, point2_name))
        self.construction_steps.append(f"Drew line connecting points {point1_name} and {point2_name}")
        return (point1_name, point2_name)
    
    def add_circle(self, center_name, point_on_circle_name):
        """Add a circle with given center and point on the circle using compass."""
        if center_name not in self.points:
            raise ValueError(f"Center point {center_name} does not exist.")
        if point_on_circle_name not in self.points:
            raise ValueError(f"Point {point_on_circle_name} does not exist.")
        
        self.circles.append((center_name, point_on_circle_name))
        
        # Calculate the radius
        center = self.points[center_name]
        point = self.points[point_on_circle_name]
        radius = math.sqrt((center[0] - point[0])**2 + (center[1] - point[1])**2)
        
        self.construction_steps.append(
            f"Drew circle with center {center_name} through point {point_on_circle_name} with radius {radius:.2f}")
        return (center_name, point_on_circle_name)
    
    def find_intersection_line_line(self, line1, line2, name=None):
        """Find the intersection of two lines and add it as a new point."""
        # Extract points
        p1_name, p2_name = line1
        p3_name, p4_name = line2
        
        p1 = np.array(self.points[p1_name])
        p2 = np.array(self.points[p2_name])
        p3 = np.array(self.points[p3_name])
        p4 = np.array(self.points[p4_name])
        
        # Check if lines are parallel
        v1 = p2 - p1
        v2 = p4 - p3
        
        cross_product = np.cross(v1, v2)
        if abs(cross_product) < 1e-10:  # Lines are parallel
            return None
        
        # Solve for intersection
        # Line 1: p1 + t*v1
        # Line 2: p3 + s*v2
        # Find t and s where p1 + t*v1 = p3 + s*v2
        
        # From vector form to parametric equations:
        # x1 + t*(x2-x1) = x3 + s*(x4-x3)
        # y1 + t*(y2-y1) = y3 + s*(y4-y3)
        
        # Solve for t:
        # t = ((x3-x1)*(y4-y3) - (y3-y1)*(x4-x3)) / ((x2-x1)*(y4-y3) - (y2-y1)*(x4-x3))
        
        denominator = (v1[0]*v2[1] - v1[1]*v2[0])
        t = ((p3[0]-p1[0])*v2[1] - (p3[1]-p1[1])*v2[0]) / denominator
        
        # Calculate intersection point
        intersection = p1 + t * v1
        
        # Add the intersection point
        if name is None:
            name = f"I_{p1_name}{p2_name}_{p3_name}{p4_name}"
        
        self.add_point(name, intersection[0], intersection[1], 
                       source=f"intersection of lines {p1_name}{p2_name} and {p3_name}{p4_name}")
        return name
    
    def find_intersection_circle_circle(self, circle1, circle2, name1=None, name2=None):
        """Find the intersections of two circles and add them as new points."""
        # Extract center points and radii
        center1_name, point1_name = circle1
        center2_name, point2_name = circle2
        
        center1 = np.array(self.points[center1_name])
        point1 = np.array(self.points[point1_name])
        center2 = np.array(self.points[center2_name])
        point2 = np.array(self.points[point2_name])
        
        radius1 = np.linalg.norm(point1 - center1)
        radius2 = np.linalg.norm(point2 - center2)
        
        # Calculate distance between centers
        d = np.linalg.norm(center2 - center1)
        
        # Check if circles are separate, contained, or coincident
        if d > radius1 + radius2 or d < abs(radius1 - radius2) or (d == 0 and radius1 == radius2):
            return None, None
        
        # Calculate intersection points
        a = (radius1**2 - radius2**2 + d**2) / (2 * d)
        h = np.sqrt(radius1**2 - a**2)
        
        # Calculate the midpoint
        p2 = center1 + a * (center2 - center1) / d
        
        # Calculate the intersection points
        fx = p2[0] + h * (center2[1] - center1[1]) / d
        fy = p2[1] - h * (center2[0] - center1[0]) / d
        gx = p2[0] - h * (center2[1] - center1[1]) / d
        gy = p2[1] + h * (center2[0] - center1[0]) / d
        
        # Add the intersection points
        if name1 is None:
            name1 = f"I1_{center1_name}{point1_name}_{center2_name}{point2_name}"
        
        if name2 is None:
            name2 = f"I2_{center1_name}{point1_name}_{center2_name}{point2_name}"
        
        source = f"intersection of circles {center1_name}{point1_name} and {center2_name}{point2_name}"
        
        self.add_point(name1, fx, fy, source=source)
        if d != abs(radius1 - radius2):  # If circles are tangent, only one intersection
            self.add_point(name2, gx, gy, source=source)
            return name1, name2
        return name1, None
    
    def find_intersection_line_circle(self, line, circle, name1=None, name2=None):
        """Find the intersections of a line and a circle and add them as new points."""
        # Extract points
        p1_name, p2_name = line
        center_name, point_name = circle
        
        p1 = np.array(self.points[p1_name])
        p2 = np.array(self.points[p2_name])
        center = np.array(self.points[center_name])
        point = np.array(self.points[point_name])
        
        radius = np.linalg.norm(point - center)
        
        # Vector from p1 to p2
        d = p2 - p1
        
        # Vector from p1 to center
        f = p1 - center
        
        # Quadratic equation coefficients
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:  # No intersection
            return None, None
        
        # Calculate intersection points
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        intersection1 = p1 + t1 * d
        intersection2 = p1 + t2 * d
        
        if discriminant == 0:  # Line is tangent to circle, only one intersection
            if name1 is None:
                name1 = f"I_{p1_name}{p2_name}_{center_name}{point_name}"
            
            self.add_point(name1, intersection1[0], intersection1[1], 
                          source=f"intersection of line {p1_name}{p2_name} and circle {center_name}{point_name}")
            return name1, None
        
        # Add the intersection points
        if name1 is None:
            name1 = f"I1_{p1_name}{p2_name}_{center_name}{point_name}"
        
        if name2 is None:
            name2 = f"I2_{p1_name}{p2_name}_{center_name}{point_name}"
        
        source = f"intersection of line {p1_name}{p2_name} and circle {center_name}{point_name}"
        
        self.add_point(name1, intersection1[0], intersection1[1], source=source)
        self.add_point(name2, intersection2[0], intersection2[1], source=source)
        
        return name1, name2
    
    def add_reasoning(self, text):
        """Add a reasoning step to document the LLM's thinking process."""
        self.reasoning_steps.append(text)
    
    def plot(self):
        """Plot the current state of the geometric construction."""
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Plot points
        for name, (x, y) in self.points.items():
            self.ax.plot(x, y, 'o', markersize=5)
            self.ax.text(x + 0.2, y + 0.2, name, fontsize=12)
        
        # Plot lines
        for p1_name, p2_name in self.lines:
            x1, y1 = self.points[p1_name]
            x2, y2 = self.points[p2_name]
            
            # For visualization, extend the line to the edges of the plot
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1e-10:  # Vertical line
                self.ax.axvline(x=x1, color='blue', linestyle='-', alpha=0.6)
            else:
                slope = dy / dx
                intercept = y1 - slope * x1
                
                x_vals = np.array([-10, 10])
                y_vals = slope * x_vals + intercept
                
                # Check if the line is within plot limits
                valid_indices = (y_vals >= -10) & (y_vals <= 10)
                if np.any(valid_indices):
                    self.ax.plot(x_vals[valid_indices], y_vals[valid_indices], 'b-', alpha=0.6)
        
        # Plot circles
        for center_name, point_name in self.circles:
            center_x, center_y = self.points[center_name]
            point_x, point_y = self.points[point_name]
            
            radius = np.sqrt((center_x - point_x)**2 + (center_y - point_y)**2)
            circle = Circle((center_x, center_y), radius, fill=False, color='green', alpha=0.6)
            self.ax.add_patch(circle)
        
        plt.draw()
        plt.pause(0.001)
    
    def save_figure(self, filename="geometry_construction.png"):
        """Save the current figure to a file."""
        self.fig.savefig(filename)
        return filename
    
    def print_reasoning(self):
        """Print all the reasoning steps."""
        print("\n=== Reasoning Steps ===")
        for i, step in enumerate(self.reasoning_steps, 1):
            print(f"Step {i}: {step}")
    
    def print_construction(self):
        """Print all the construction steps."""
        print("\n=== Construction Steps ===")
        for i, step in enumerate(self.construction_steps, 1):
            print(f"Step {i}: {step}")
    
    def perpendicular_bisector(self, point1_name, point2_name):
        """Construct the perpendicular bisector of a line segment."""
        # Get the two points
        p1 = np.array(self.points[point1_name])
        p2 = np.array(self.points[point2_name])
        
        # Draw circles of equal radius from both points
        circle1 = self.add_circle(point1_name, point2_name)
        circle2 = self.add_circle(point2_name, point1_name)
        
        # Find the intersection points
        int1, int2 = self.find_intersection_circle_circle(circle1, circle2)
        
        if int1 and int2:
            # Draw the perpendicular bisector
            line = self.add_line(int1, int2)
            return line
        return None
    
    def angle_bisector(self, vertex_name, point1_name, point2_name):
        """Construct the angle bisector of an angle defined by three points."""
        # Get the three points
        vertex = np.array(self.points[vertex_name])
        p1 = np.array(self.points[point1_name])
        p2 = np.array(self.points[point2_name])
        
        # Draw circles of equal radius centered at the vertex
        midpoint_name = f"M_{vertex_name}"
        
        # Get vectors from vertex to the two points
        v1 = p1 - vertex
        v2 = p2 - vertex
        
        # Normalize the vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Create a point at equal distance from the vertex along each ray
        dist = 3.0  # arbitrary distance
        eq_p1_name = f"Eq1_{vertex_name}"
        eq_p2_name = f"Eq2_{vertex_name}"
        
        self.add_point(eq_p1_name, vertex[0] + dist * v1_norm[0], vertex[1] + dist * v1_norm[1],
                      source=f"point at distance {dist} from {vertex_name} along {point1_name}")
        self.add_point(eq_p2_name, vertex[0] + dist * v2_norm[0], vertex[1] + dist * v2_norm[1],
                      source=f"point at distance {dist} from {vertex_name} along {point2_name}")
        
        # Draw circles centered at these equal distance points
        radius = dist * 0.8  # smaller than the distance to avoid issues
        
        # Create points at a distance 'radius' from the equal distance points
        rad_p1_name = f"Rad1_{vertex_name}"
        rad_p2_name = f"Rad2_{vertex_name}"
        
        self.add_point(rad_p1_name, 
                      self.points[eq_p1_name][0] + radius, 
                      self.points[eq_p1_name][1],
                      source=f"point at radius {radius} from {eq_p1_name}")
        
        self.add_point(rad_p2_name, 
                      self.points[eq_p2_name][0] + radius, 
                      self.points[eq_p2_name][1],
                      source=f"point at radius {radius} from {eq_p2_name}")
        
        circle1 = self.add_circle(eq_p1_name, rad_p1_name)
        circle2 = self.add_circle(eq_p2_name, rad_p2_name)
        
        # Find the intersection points
        int1, int2 = self.find_intersection_circle_circle(circle1, circle2)
        
        if int1:
            # Draw the angle bisector
            bisector = self.add_line(vertex_name, int1)
            return bisector
        return None
    
    def parallel_line(self, line, point_name):
        """Construct a line through a point parallel to a given line."""
        p1_name, p2_name = line
        
        # Get the points
        p1 = np.array(self.points[p1_name])
        p2 = np.array(self.points[p2_name])
        point = np.array(self.points[point_name])
        
        # Direction vector of the original line
        direction = p2 - p1
        
        # Create a new point in the parallel direction
        new_point_name = f"P_{point_name}_{p1_name}{p2_name}"
        self.add_point(new_point_name, 
                      point[0] + direction[0], 
                      point[1] + direction[1],
                      source=f"point in direction of {p1_name}{p2_name} from {point_name}")
        
        # Draw the parallel line
        parallel = self.add_line(point_name, new_point_name)
        return parallel
    
    def perpendicular_line(self, line, point_name):
        """Construct a line through a point perpendicular to a given line."""
        p1_name, p2_name = line
        
        # Get the points
        p1 = np.array(self.points[p1_name])
        p2 = np.array(self.points[p2_name])
        point = np.array(self.points[point_name])
        
        # Create a circle centered at the point
        aux_point_name = f"Aux_{point_name}"
        self.add_point(aux_point_name, 
                      point[0] + 5, 
                      point[1],
                      source=f"auxiliary point for perpendicular construction")
        
        circle = self.add_circle(point_name, aux_point_name)
        
        # Find intersections of the circle with the line
        int1, int2 = self.find_intersection_line_circle(line, circle)
        
        if int1 and int2:
            # Create the perpendicular bisector of the segment between intersections
            return self.perpendicular_bisector(int1, int2)
        elif int1:
            # Line is tangent to circle, create perpendicular at tangent point
            return self.add_line(point_name, int1)
        
        return None

# Example usage
def demo_geometry_notebook():
    notebook = GeometryNotebook()
    
    # Example: Construct an equilateral triangle
    notebook.add_reasoning("I will construct an equilateral triangle using compass and straightedge.")
    
    # Create two initial points for the base
    notebook.add_point("A", -3, 0)
    notebook.add_point("B", 3, 0)
    
    # Create a line segment AB
    notebook.add_line("A", "B")
    
    # Create circles centered at A and B with radius AB
    notebook.add_circle("A", "B")
    notebook.add_circle("B", "A")
    
    # Find the intersection point of the circles (the third vertex of the triangle)
    C, _ = notebook.find_intersection_circle_circle(
        ("A", "B"), 
        ("B", "A"), 
        name1="C"
    )
    
    # Complete the triangle
    notebook.add_line("A", "C")
    notebook.add_line("B", "C")
    
    notebook.add_reasoning("The equilateral triangle ABC is now constructed.")
    notebook.add_reasoning("All sides are equal because each has length equal to the original segment AB.")
    
    # Plot the construction
    notebook.plot()
    notebook.print_reasoning()
    notebook.print_construction()
    notebook.save_figure("equilateral_triangle.png")
    
    return notebook

if __name__ == "__main__":
    demo_geometry_notebook()