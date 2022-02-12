import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:
    reader = csv.DictReader(csv_fh)
    for row in reader:
        years.append(float(row['Year']))
        sunspot_counts.append(float(row['Sunspot_Count']))
        republican_counts.append(float(row['Republican_Count']))

years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
def make_basis(x, part = 'a', is_years = True):
    assert(part in ['a', 'b', 'c', 'd'])
    if part == 'a':
        scaled = ((x - 1960) / 40) if is_years else (x / 20)
        basis = [scaled ** i for i in range(1, 6)]
    if part == 'b':
        assert(is_years)
        basis = [np.exp((x - i) ** 2 / -25) for i in range(1960, 2011, 5)]
    if part == 'c':
        basis = [np.cos(x / i) for i in range(1, 6)]
    if part == 'd':
        basis = [np.cos(x / i) for i in range(1, 26)]

    # add basis for bias and stack all basis vectors
    return np.vstack((np.ones(x.shape), *basis)).T

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X, Y):
    return np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))

def compute_loss(Y, Yhat):
    return np.sum((Y - Yhat) ** 2)

if __name__ == '__main__':
    parts = ['a', 'b', 'c', 'd']
    grid_years = np.linspace(1960, 2005, 200)
    for part in parts:
        years_basis = make_basis(years, part = part)
        w = find_weights(years_basis, republican_counts)
        loss = compute_loss(republican_counts, np.dot(years_basis, w))
        print(f"Loss for Years (part {part}): {loss}")
        grid_Yhat  = np.dot(make_basis(grid_years, part = part), w)
        # Plot the data and the regression line.
        plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
        plt.xlabel("Year")
        plt.ylabel("Number of Republicans in Congress")
        plt.show()

    sunspot_counts = sunspot_counts[years<last_year]
    republican_counts = republican_counts[years<last_year]
    grid_sunspots = np.linspace(0, 175, 800)
    for part in parts:
        if part == 'b':
            continue
        sunspots_basis = make_basis(sunspot_counts, part = part, is_years = False)
        w = find_weights(sunspots_basis, republican_counts)
        loss = compute_loss(republican_counts, np.dot(sunspots_basis, w))
        print(f"Loss for Sunspots (part {part}): {loss}")
        grid_Yhat  = np.dot(make_basis(grid_sunspots, part = part, is_years = False), w)
        # Plot the data and the regression line.
        plt.plot(sunspot_counts, republican_counts, 'o', grid_sunspots, grid_Yhat, '-')
        plt.ylim([0, 200])
        plt.xlabel("Number of Sunspots")
        plt.ylabel("Number of Republicans in Congress")
        plt.show()