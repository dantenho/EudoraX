
import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import fs from 'fs';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

const USERS_FILE = path.join(process.cwd(), 'users.json');
const DATA_FILE = path.join(process.cwd(), 'data.json');
const JWT_SECRET = process.env.JWT_SECRET || 'eudorax-ultra-secret-2026';

// Initialize users file if not exists
if (!fs.existsSync(USERS_FILE)) {
  fs.writeFileSync(USERS_FILE, JSON.stringify([]));
}

// Initialize data file if not exists
if (!fs.existsSync(DATA_FILE)) {
  fs.writeFileSync(DATA_FILE, JSON.stringify({ folders: [], assets: [] }));
}

function getUsers() {
  return JSON.parse(fs.readFileSync(USERS_FILE, 'utf8'));
}

function saveUsers(users: any[]) {
  fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2));
}

function getData() {
  return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
}

function saveData(data: any) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
}

// Middleware to verify JWT
const authenticateToken = (req: any, res: any, next: any) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) return res.sendStatus(401);

  jwt.verify(token, JWT_SECRET, (err: any, user: any) => {
    if (err) return res.sendStatus(403);
    req.user = user;
    next();
  });
};

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Routes
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', version: '4.8.0' });
  });

  // Auth Routes
  app.post('/api/auth/signup', async (req, res) => {
    const { email, password, name } = req.body;
    const users = getUsers();
    
    if (users.find((u: any) => u.email === email)) {
      return res.status(400).json({ error: 'User already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = { id: Date.now().toString(), email, password: hashedPassword, name };
    users.push(newUser);
    saveUsers(users);

    const token = jwt.sign({ id: newUser.id, email: newUser.email }, JWT_SECRET);
    res.json({ token, user: { id: newUser.id, email: newUser.email, name: newUser.name } });
  });

  app.post('/api/auth/login', async (req, res) => {
    const { email, password } = req.body;
    const users = getUsers();
    const user = users.find((u: any) => u.email === email);

    if (!user || !(await bcrypt.compare(password, user.password))) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET);
    res.json({ token, user: { id: user.id, email: user.email, name: user.name, isVerified: user.isVerified || false } });
  });

  // Profile Management
  app.get('/api/auth/profile', authenticateToken, (req: any, res) => {
    const users = getUsers();
    const user = users.find((u: any) => u.id === req.user.id);
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json({ id: user.id, email: user.email, name: user.name, isVerified: user.isVerified || false });
  });

  app.put('/api/auth/profile', authenticateToken, async (req: any, res) => {
    const { name, email, password } = req.body;
    const users = getUsers();
    const userIndex = users.findIndex((u: any) => u.id === req.user.id);
    if (userIndex === -1) return res.status(404).json({ error: 'User not found' });

    if (name) users[userIndex].name = name;
    if (email) users[userIndex].email = email;
    if (password) {
      users[userIndex].password = await bcrypt.hash(password, 10);
    }

    saveUsers(users);
    res.json({ message: 'Profile updated successfully', user: { id: users[userIndex].id, email: users[userIndex].email, name: users[userIndex].name } });
  });

  // Password Reset (Mock)
  app.post('/api/auth/reset-password', (req, res) => {
    const { email } = req.body;
    // In a real app, send an email with a token
    res.json({ message: `Password reset link sent to ${email} (Mocked)` });
  });

  // Email Verification (Mock)
  app.post('/api/auth/verify-email', authenticateToken, (req: any, res) => {
    const users = getUsers();
    const userIndex = users.findIndex((u: any) => u.id === req.user.id);
    if (userIndex === -1) return res.status(404).json({ error: 'User not found' });

    users[userIndex].isVerified = true;
    saveUsers(users);
    res.json({ message: 'Email verified successfully (Mocked)' });
  });

  // Asset Organization
  app.get('/api/folders', authenticateToken, (req: any, res) => {
    const data = getData();
    const userFolders = data.folders.filter((f: any) => f.userId === req.user.id);
    res.json(userFolders);
  });

  app.post('/api/folders', authenticateToken, (req: any, res) => {
    const { name } = req.body;
    const data = getData();
    const newFolder = { id: Date.now().toString(), name, userId: req.user.id };
    data.folders.push(newFolder);
    saveData(data);
    res.json(newFolder);
  });

  app.get('/api/assets', authenticateToken, (req: any, res) => {
    const data = getData();
    const userAssets = data.assets.filter((a: any) => a.userId === req.user.id);
    res.json(userAssets);
  });

  app.post('/api/assets', authenticateToken, (req: any, res) => {
    const { name, url, tags, folderId } = req.body;
    const data = getData();
    const newAsset = { 
      id: Date.now().toString(), 
      name, 
      url, 
      tags: tags || [], 
      folderId: folderId || null, 
      userId: req.user.id,
      createdAt: new Date().toISOString()
    };
    data.assets.push(newAsset);
    saveData(data);
    res.json(newAsset);
  });

  app.put('/api/assets/:id', authenticateToken, (req: any, res) => {
    const { id } = req.params;
    const { tags, folderId, name } = req.body;
    const data = getData();
    const assetIndex = data.assets.findIndex((a: any) => a.id === id && a.userId === req.user.id);
    if (assetIndex === -1) return res.status(404).json({ error: 'Asset not found' });

    if (tags) data.assets[assetIndex].tags = tags;
    if (folderId !== undefined) data.assets[assetIndex].folderId = folderId;
    if (name) data.assets[assetIndex].name = name;

    saveData(data);
    res.json(data.assets[assetIndex]);
  });

  app.delete('/api/assets/:id', authenticateToken, (req: any, res) => {
    const { id } = req.params;
    const data = getData();
    data.assets = data.assets.filter((a: any) => !(a.id === id && a.userId === req.user.id));
    saveData(data);
    res.sendStatus(204);
  });

  // Vite integration
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`EudoraX Server running on http://localhost:${PORT}`);
  });
}

startServer();
