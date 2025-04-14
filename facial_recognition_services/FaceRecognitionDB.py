import pickle

import psycopg2


class FaceRecognitionDB:
    def __init__(self, db_config=None):
        """Initialize the face recognition system with PostgreSQL database support"""
        if db_config is None:
            self.db_config = {
                'dbname': 'face_recognition',
                'user': 'postgres',
                'password': 'mypassword',
                'host': 'localhost',
                'port': '5432'
            }
        else:
            self.db_config = db_config

        self.initialize_database()

    def get_connection(self):
        """Create and return a database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)

            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            raise

    def initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            # Connect to PostgreSQL
            conn = self.get_connection()
            cursor = conn.cursor()

            # First make sure pgvector extension is installed
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # For pgvector, we'll use the vector data type
            # Assuming face_recognition generates 128-dimensional vectors
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                id SERIAL PRIMARY KEY,
                person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                encoding BYTEA NOT NULL,
                vector_encoding vector(128),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Create an index for faster face similarity searches
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS face_encodings_person_id_idx ON face_encodings (person_id);
            ''')

            # Create a function to convert BYTEA to vector
            # This will be useful when we want to use the vector similarity functions
            cursor.execute('''
            CREATE OR REPLACE FUNCTION bytea_to_vector(bytea_data BYTEA) RETURNS vector
            AS $$
            DECLARE
                result vector;
            BEGIN
                -- Implement this function based on your serialization format
                -- For now, we'll use the vector_encoding column directly
                RETURN NULL;
            END;
            $$ LANGUAGE plpgsql;
            ''')

            conn.commit()
            cursor.close()
            conn.close()
            print("PostgreSQL database initialized")
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def add_person(self, name):
        """Add a new person to the database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Check if person already exists
            cursor.execute("SELECT id FROM persons WHERE name = %s", (name,))
            existing = cursor.fetchone()

            if existing:
                person_id = existing[0]
                print(f"Person '{name}' already exists with ID {person_id}")
            else:
                cursor.execute("INSERT INTO persons (name) VALUES (%s) RETURNING id", (name,))
                person_id = cursor.fetchone()[0]
                print(f"Added new person '{name}' with ID {person_id}")

            conn.commit()
            cursor.close()
            conn.close()
            return person_id
        except Exception as e:
            print(f"Error adding person to database: {e}")
            return None

    def add_face_encoding(self, person_id, encoding):
        """Add a face encoding for a person"""
        try:
            # Serialize the encoding (numpy array) to binary
            encoding_binary = pickle.dumps(encoding)

            # Convert to a list for direct vector storage
            encoding_list = encoding.tolist()

            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO face_encodings (person_id, encoding, vector_encoding) VALUES (%s, %s, %s) RETURNING id",
                (person_id, psycopg2.Binary(encoding_binary), encoding_list)
            )

            encoding_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            print(f"Added face encoding {encoding_id} for person ID {person_id}")
            return encoding_id
        except Exception as e:
            print(f"Error adding face encoding to database: {e}")
            return None

    def get_all_face_encodings(self):
        """Retrieve all face encodings with person names"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT p.name, f.encoding 
                FROM face_encodings f
                JOIN persons p ON f.person_id = p.id
            """)

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            # Deserialize the encodings
            known_names = []
            known_encodings = []

            for name, encoding_binary in results:
                encoding = pickle.loads(encoding_binary)
                known_names.append(name)
                known_encodings.append(encoding)

            print(f"Retrieved {len(known_encodings)} face encodings from database")
            return known_names, known_encodings
        except Exception as e:
            print(f"Error retrieving face encodings from database: {e}")
            return [], []

    def find_similar_faces(self, query_encoding, threshold=0.6):
        """Find similar faces using vector similarity search

        Args:
            query_encoding: Face encoding to search for
            threshold: Similarity threshold (lower = more similar)

        Returns:
            List of (name, similarity) tuples
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Convert query to a list
            query_list = query_encoding.tolist()

            # PostgreSQL with pgvector can do native vector similarity searches
            # Lower cosine distance = more similar
            cursor.execute("""
                SELECT p.name, 1 - (f.vector_encoding <-> %s::vector) as similarity
                FROM face_encodings f
                JOIN persons p ON f.person_id = p.id
                WHERE 1 - (f.vector_encoding <-> %s::vector) >= %s
                ORDER BY similarity DESC
            """, (query_list, query_list, 1 - threshold))

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            return results
        except Exception as e:
            print(f"Error searching for similar faces: {e}")
            return []

    def delete_person(self, name):
        """Delete a person and their face encodings"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get person ID
            cursor.execute("SELECT id FROM persons WHERE name = %s", (name,))
            result = cursor.fetchone()

            if result:
                person_id = result[0]

                # Delete the person (will cascade to face_encodings)
                cursor.execute("DELETE FROM persons WHERE id = %s", (person_id,))

                conn.commit()
                print(f"Deleted person '{name}' and all associated face encodings")
                success = True
            else:
                print(f"Person '{name}' not found in database")
                success = False

            cursor.close()
            conn.close()
            return success
        except Exception as e:
            print(f"Error deleting person from database: {e}")
            return False