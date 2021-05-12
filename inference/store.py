class Store:

    def __init__(self):
        self.W = {}
        self.b = {}
        self.dW = {}
        self.db = {}
        self.A = {}
        self.Z = {}

    # weight matrices
    def get_W(self, l):
        return self.W[l]

    def set_W(self, W_instance, l):
        self.W[l] = W_instance

    # bias vectors
    def get_b(self, l):
        return self.b[l]

    def set_b(self, b_instance, l):
        self.b[l] = b_instance

    # weight derivatives
    def get_dW(self, l):
        return self.dW[l]

    def set_dW(self, dW_instance, l):
        self.dW[l] = dW_instance

    # bias dervatives
    def get_db(self, l):
        return self.db[l]

    def set_db(self, db_instance, l):
        self.db[l] = db_instance

    # activation output vector
    def get_A(self, l):
        return self.A[l]

    def set_A(self, A_instance, l):
        self.A[l] = A_instance

    # activation input vector
    def get_Z(self, l):
        return self.Z[l]

    def set_Z(self, Z_instance, l):
        self.Z[l] = Z_instance
