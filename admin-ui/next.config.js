/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/health',
        destination: 'http://localhost:8080/health',
      },
      {
        source: '/api/ready',
        destination: 'http://localhost:8080/ready',
      },
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/admin/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
