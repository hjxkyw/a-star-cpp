// a_star.cpp
// --------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <format>
#include <memory>
#include <algorithm>
#include <chrono>
#include <thread>
#include <random>
#include <fstream>

// --------------------------------------------------------------------------------

// Class: Location
// Represents a physical 2D coordinate (x, y) on the map.
struct Location
{
  int x;
  int y;

  bool operator==(const Location& other) const
  {
    return x == other.x && y == other.y;
  }

  // Returns a unique string key used for tracking explored locations in hashes.
  std::string key() const
  {
    return std::format("{},{}", x, y);
  }

  // Manhattan Distance: The sum of the absolute differences of their coordinates.
  int dist(const Location& p) const
  {
    return std::abs(x - p.x) + std::abs(y - p.y);
  }

  std::string to_string() const
  {
    return std::format("(Col {}, Row {})", x, y);
  }
};

// Custom hash for Location to use in unordered_map (if needed directly)
struct LocationHash
{
  std::size_t operator()(const Location& loc) const
  {
    return std::hash<std::string>{}(loc.key());
  }
};

// Class: Node
// A Node represents a specific "promise" in the search tree.
struct Node
{
  Location location;
  std::shared_ptr<Node> parent;
  std::string action;
  double g_cost = 0;
  double h_cost = 0;

  double f_cost() const
  {
    return g_cost + h_cost;
  }
};

// Comparator for the priority queue (Min-Heap)
struct NodePtrCompare
{
  bool operator()(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) const
  {
    return a->f_cost() > b->f_cost();
  }
};

// Class: TerrainMap
// Manages the grid environment, terrain costs, and the visual rendering logic.
class TerrainMap
{
public:
  static constexpr int COST_GRASS = 1;
  static constexpr int COST_MUD = 10;

  int width;
  int height;
  Location start;
  Location goal;
  std::unordered_map<std::string, bool> mud_tiles;

  TerrainMap(int w, int h, Location s, Location g)
    : width(w), height(h), start(s), goal(g)
  {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int x = 0; x < width; ++x)
    {
      for (int y = 0; y < height; ++y)
      {
        Location p{x, y};
        if (p == start || p == goal) continue;
        if (dis(gen) < 0.3)
        {
          mud_tiles[p.key()] = true;
        }
      }
    }
  }

  bool is_goal(const Location& loc) const { return loc == goal; }
  double heuristic(const Location& loc) const { return loc.dist(goal); }
  bool is_mud(const Location& loc) const { return mud_tiles.contains(loc.key()); }

  struct Move { std::string action; Location loc; int cost; };

  std::vector<Move> successors(const Location& loc) const
  {
    std::vector<Move> moves;
    // Directions: Down, Up, Right, Left
    const int dx[] = {0, 0, 1, -1};
    const int dy[] = {1, -1, 0, 0};
    const std::string labels[] = {"Down", "Up", "Right", "Left"};

    for (int i = 0; i < 4; ++i)
    {
      int nx = loc.x + dx[i];
      int ny = loc.y + dy[i];

      if (nx >= 0 && nx < width && ny >= 0 && ny < height)
      {
        Location p{nx, ny};
        int cost = is_mud(p) ? COST_MUD : COST_GRASS;
        moves.push_back({labels[i], p, cost});
      }
    }
    return moves;
  }

  void render(const Location& agent_loc, const std::vector<std::string>& path_actions,
              const std::vector<std::shared_ptr<Node>>& frontier,
              const std::unordered_map<std::string, bool>& closed_set, std::ostream& out) const
  {
    std::unordered_map<std::string, std::string> path_keys;
    Location curr = start;

    for (const auto& act : path_actions)
    {
      if (act == "Up") curr.y--;
      else if (act == "Down") curr.y++;
      else if (act == "Left") curr.x--;
      else if (act == "Right") curr.x++;

      std::string m;
      if (is_mud(curr))
      {
        if (act == "Up") m = " ⬆ ";
        else if (act == "Down") m = " ⬇ ";
        else if (act == "Left") m = " ⬅ ";
        else m = " ➡ ";
      }
      else
      {
        if (act == "Up") m = " ↑ ";
        else if (act == "Down") m = " ↓ ";
        else if (act == "Left") m = " ← ";
        else m = " → ";
      }

      if (curr.key() != agent_loc.key()) path_keys[curr.key()] = m;
    }

    std::unordered_map<std::string, bool> frontier_keys;
    for (const auto& node : frontier)
    {
      if (node->location.key() != agent_loc.key() && !path_keys.contains(node->location.key()))
      {
        frontier_keys[node->location.key()] = true;
      }
    }

    out << "\n" << std::string(30, '=') << "\nA* GRID STATE\n" << std::string(30, '-') << "\n";

    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        Location p{x, y};
        std::string k = p.key();

        if (k == agent_loc.key()) out << " ● ";
        else if (k == goal.key()) out << " ★ ";
        else if (k == start.key()) out << " ○ ";
        else if (path_keys.contains(k)) out << path_keys.at(k);
        else if (frontier_keys.contains(k)) out << (is_mud(p) ? " M " : " F ");
        else if (closed_set.contains(k)) out << " - ";
        else if (is_mud(p)) out << " ~ ";
        else out << " . ";
      }
      out << "\n";
    }
    out << std::string(30, '-') << "\n";
  }
};

// --------------------------------------------------------------------------------

std::vector<std::string> reconstruct_path(std::shared_ptr<Node> end_node)
{
  std::vector<std::string> path;
  while (end_node && end_node->parent)
  {
    path.insert(path.begin(), end_node->action);
    end_node = end_node->parent;
  }
  return path;
}

void run_experiment(TerrainMap& map, std::ostream& out, bool interactive, bool auto_play = false, double wait_time = 1.0, bool show_menu = true)
{
  auto root = std::make_shared<Node>(map.start, nullptr, "", 0.0, map.heuristic(map.start));

  std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodePtrCompare> frontier;
  frontier.push(root);

  std::unordered_map<std::string, double> best;
  best[map.start.key()] = 0.0;

  std::unordered_map<std::string, bool> closed_set;
  int step_count = 0;

  while (!frontier.empty())
  {
    auto curr = frontier.top();
    frontier.pop();
    step_count++;

    closed_set[curr->location.key()] = true;
    std::string terrain = map.is_mud(curr->location) ? "(MUD)" : "(GRASS)";
    std::string act_text = !curr->action.empty() ? "Moved " + curr->action : "Started";

    out << std::format("\nSTEP {}:\n  CHOSEN: {} to {} {}\n", step_count, act_text, curr->location.to_string(), terrain);
    out << std::format("  COST DETAIL: f={} (g={} + h={})\n", curr->f_cost(), curr->g_cost, curr->h_cost);

    if (map.is_goal(curr->location))
    {
      auto final_path = reconstruct_path(curr);

      std::vector<std::shared_ptr<Node>> frontier_list;
      auto temp_q = frontier;
      while(!temp_q.empty()) { frontier_list.push_back(temp_q.top()); temp_q.pop(); }

      map.render(curr->location, final_path, frontier_list, closed_set, out);

      double path_length = static_cast<double>(final_path.size());
      double total_explored = static_cast<double>(closed_set.size());
      double efficiency = total_explored > 0 ? (path_length / total_explored) * 100.0 : 0.0;

      out << std::format("\nSUCCESS! Goal reached at {}.\n", curr->location.to_string());
      out << "--------------------------------------------------\nEXPLORATION DIAGNOSTICS:\n";
      out << std::format("  Path Length:       {} steps\n", path_length);
      out << std::format("  Total Explored:    {} points\n", total_explored);
      out << std::format("  Search Efficiency: {:.2f}%\n", efficiency);
      out << "--------------------------------------------------\n";
      return;
    }

    double known_best_g = best.contains(curr->location.key()) ? best[curr->location.key()] : 1e18;

    if (curr->g_cost <= known_best_g)
    {
      for (const auto& t : map.successors(curr->location))
      {
        double new_g = curr->g_cost + t.cost;
        std::string key = t.loc.key();

        if (!best.contains(key) || new_g < best[key])
        {
          best[key] = new_g;
          frontier.push(std::make_shared<Node>(t.loc, curr, t.action, new_g, map.heuristic(t.loc)));
        }
      }
    }

    std::vector<std::shared_ptr<Node>> frontier_list;
    auto temp_q = frontier;
    while(!temp_q.empty()) { frontier_list.push_back(temp_q.top()); temp_q.pop(); }

    map.render(curr->location, reconstruct_path(curr), frontier_list, closed_set, out);
    out << std::format("CURRENT LOCATION: {} {}\n", curr->location.to_string(), terrain);

    if (show_menu)
    {
      out << "  MENU FOR NEXT STEP (Current Frontier):\n";
      if (!frontier_list.empty())
      {
        std::sort(frontier_list.begin(), frontier_list.end(), [](const auto& a, const auto& b) {
          return a->f_cost() < b->f_cost();
        });

        bool first = true;
        for (const auto& n : frontier_list)
        {
          std::string marker = first ? " -> " : "    ";
          out << std::format("{} {}: f={} (g={} + h={})\n", marker, n->location.to_string(), n->f_cost(), n->g_cost, n->h_cost);
          first = false;
        }
      }
      else
      {
        out << "    (Empty)\n";
      }
    }

    if (interactive)
    {
      if (auto_play)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(wait_time * 1000)));
      }
      else
      {
        std::cout << "\n[Press ENTER for next step...]";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
    }
  }
}

int main(int argc, char* argv[])
{
  TerrainMap map(10, 10, {0, 0}, {9, 9});

  if (argc == 1)
  {
    run_experiment(map, std::cout, true, false, 1.0, true);
  }
  else
  {
    std::string arg = argv[1];

    // Check if argument is a flag
    if (arg[0] == '-')
    {
      if (arg == "-")
      {
        std::cout << "Auto-play mode (1.0s delay)...\n";
        run_experiment(map, std::cout, true, true, 1.0, false);
      }
      else
      {
        try
        {
          double w = std::stod(arg.substr(1));
          std::cout << std::format("Auto-play mode ({}s delay)...\n", w);
          run_experiment(map, std::cout, true, true, w, false);
        }
        catch (...)
        {
          std::cout << "Invalid delay argument. Using default.\n";
          run_experiment(map, std::cout, true, true, 1.0, false);
        }
      }
    }
    // Argument is a filename
    else
    {
      std::ofstream outfile(arg);
      if (outfile.is_open())
      {
        std::cout << std::format("Logging output to file: {} ...\n", arg);
        // interactive=false, auto_play=false (ignored), wait=0, show_menu=false
        run_experiment(map, outfile, false, false, 0.0, false);
        std::cout << "Done.\n";
      }
      else
      {
        std::cerr << std::format("Error: Could not open file '{}' for writing.\n", arg);
        return 1;
      }
    }
  }

  return 0;
}

// --------------------------------------------------------------------------------
// the end
