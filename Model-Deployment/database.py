import sqlite3
import os

DATABASE = "detections.db"


class Database:
    """Database handler for detections and alerts"""

    def __init__(self):
        self.db_path = DATABASE

    def init_db(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT UNIQUE,
                    first_detected DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_compliant BOOLEAN DEFAULT 0,
                    missing_ppe TEXT,
                    detected_ppe TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT,
                    snapshot_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(instance_id) REFERENCES instances(instance_id)
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT,
                    description TEXT,
                    snapshot_path TEXT
                )
            """)

            conn.commit()
            print("Database file:", os.path.abspath(self.db_path))

            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            print(c.fetchall())
            conn.close()

            print("Database initialized successfully")

        except Exception as e:
            print(f"Error initializing database: {e}")

    def log_instance_snapshot(self, instance_id, missing_ppe, detected_ppe, snapshot_path):
        try:
            if not instance_id or not snapshot_path:
                return False

            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute(
                "SELECT id FROM instances WHERE instance_id=?",
                (instance_id,)
            )

            if not c.fetchone():
                c.execute("""
                    INSERT INTO instances
                    (instance_id, is_compliant, missing_ppe, detected_ppe)
                    VALUES (?, ?, ?, ?)
                """, (
                    instance_id,
                    False,
                    ",".join(missing_ppe),
                    ",".join(detected_ppe)
                ))
            else:
                c.execute("""
                    UPDATE instances
                    SET last_updated=CURRENT_TIMESTAMP,
                        missing_ppe=?,
                        detected_ppe=?
                    WHERE instance_id=?
                """, (
                    ",".join(missing_ppe),
                    ",".join(detected_ppe),
                    instance_id
                ))

            c.execute("""
                INSERT INTO snapshots(instance_id, snapshot_path)
                VALUES(?, ?)
            """, (
                instance_id,
                snapshot_path
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error logging instance snapshot: {e}")
            return False

    def log_alert(self, alert_type, description, snapshot_path):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute("""
                INSERT INTO alerts(alert_type, description, snapshot_path)
                VALUES (?, ?, ?)
            """, (
                alert_type,
                description,
                snapshot_path
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error logging alert: {e}")

    def get_statistics(self):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute("SELECT COUNT(*) FROM instances WHERE is_compliant=0")
            non_compliant = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = c.fetchone()[0]

            conn.close()

            return {
                "total_detections": non_compliant,
                "non_compliant_count": non_compliant,
                "total_alerts": total_alerts
            }

        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                "total_detections": 0,
                "non_compliant_count": 0,
                "total_alerts": 0
            }

    def get_all_instances(self, sort_by="first_detected", sort_order="desc"):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            query = f"""
                SELECT i.*, COUNT(s.id)
                FROM instances i
                LEFT JOIN snapshots s
                ON i.instance_id=s.instance_id
                WHERE i.is_compliant=0
                GROUP BY i.instance_id
                ORDER BY i.{sort_by} {sort_order}
            """

            c.execute(query)
            rows = c.fetchall()

            conn.close()

            instances = []

            for row in rows:
                instances.append({
                    "id": row[0],
                    "instance_id": row[1],
                    "first_detected": row[2],
                    "last_updated": row[3],
                    "is_compliant": bool(row[4]),
                    "missing_ppe": row[5].split(",") if row[5] else [],
                    "detected_ppe": row[6].split(",") if row[6] else [],
                    "snapshot_count": row[7]
                })

            return instances

        except Exception as e:
            print(f"Error getting instances: {e}")
            return []

    def get_instance_snapshots(self, instance_id):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute(
                "SELECT * FROM instances WHERE instance_id=?",
                (instance_id,)
            )

            instance = c.fetchone()

            if not instance:
                conn.close()
                return None

            c.execute("""
                SELECT snapshot_path,timestamp
                FROM snapshots
                WHERE instance_id=?
                ORDER BY timestamp ASC
            """, (
                instance_id,
            ))

            snapshots = c.fetchall()

            conn.close()

            return {
                "instance_id": instance[1],
                "first_detected": instance[2],
                "last_updated": instance[3],
                "missing_ppe": instance[5].split(",") if instance[5] else [],
                "detected_ppe": instance[6].split(",") if instance[6] else [],
                "snapshots": [
                    {
                        "path": s[0],
                        "timestamp": s[1]
                    } for s in snapshots
                ]
            }

        except Exception as e:
            print(f"Error getting snapshots: {e}")
            return None

    def delete_instance(self, instance_id):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute(
                "SELECT snapshot_path FROM snapshots WHERE instance_id=?",
                (instance_id,)
            )

            for row in c.fetchall():
                if row[0] and os.path.exists(row[0]):
                    os.remove(row[0])

            c.execute(
                "DELETE FROM snapshots WHERE instance_id=?",
                (instance_id,)
            )

            c.execute(
                "DELETE FROM instances WHERE instance_id=?",
                (instance_id,)
            )

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"Error deleting instance: {e}")
            return False